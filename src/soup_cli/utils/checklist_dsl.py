"""v0.65.0 Part D — CheckList behavioural DSL.

Three test kinds from the CheckList paper (Ribeiro et al. 2020):

- ``mft`` — Minimum Functionality Test. Expects an answer keyword to appear
  in the response.
- ``inv`` — Invariance. The same answer must appear for all paraphrases of
  the prompt.
- ``dir`` — Directional Expectation. Response must shift in a known direction
  when a known perturbation is applied (e.g. negation).

Specs are YAML files with shape::

    tests:
      - name: capital-france
        kind: mft
        prompts: [What is the capital of France?]
        expected: [paris]
      - name: paraphrase-add
        kind: inv
        prompts:
          - Add 2 and 2.
          - Add two and two.

Operator-supplied responses are passed in via the ``evidence`` mapping:
``{test_name: [response_for_prompt_1, response_for_prompt_2, ...]}``. If
``evidence`` is None or a test has no entry, the test renders a neutral
``OK`` verdict (matches v0.56 / v0.61 evidence-loader policy).
"""
from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import yaml

# Closed allowlist.
CHECKLIST_KINDS = frozenset({"mft", "inv", "dir"})

# Verdict allowlist (mirrors v0.26 / v0.56 / v0.65 Part B taxonomy).
_VERDICTS = frozenset({"OK", "MINOR", "MAJOR"})

# DoS / sanity caps.
_MAX_TESTS = 1000
_MAX_PROMPTS_PER_TEST = 10_000
_MAX_EXPECTED_PER_TEST = 1000
_MAX_NAME_LEN = 128
_MAX_PROMPT_LEN = 8 * 1024
_MAX_EXPECTED_LEN = 1024
_MAX_FILE_BYTES = 1 * 1024 * 1024  # 1 MiB


def validate_test_kind(kind: object) -> str:
    """Validate a CheckList test kind. Case-insensitive."""
    if isinstance(kind, bool):
        raise TypeError("kind must be str, got bool")
    if not isinstance(kind, str):
        raise TypeError(f"kind must be str, got {type(kind).__name__}")
    if "\x00" in kind:
        raise ValueError("kind must not contain null bytes")
    if not kind:
        raise ValueError("kind must not be empty")
    canonical = kind.strip().lower()
    if canonical not in CHECKLIST_KINDS:
        raise ValueError(
            f"unknown kind {canonical!r}; valid: {sorted(CHECKLIST_KINDS)}"
        )
    return canonical


def _validate_name(name: object, *, field: str) -> str:
    if not isinstance(name, str):
        raise ValueError(f"{field} must be str")
    if "\x00" in name:
        raise ValueError(f"{field} must not contain null bytes")
    if not name:
        raise ValueError(f"{field} must not be empty")
    if len(name) > _MAX_NAME_LEN:
        raise ValueError(f"{field} too int")
    return name


def _validate_string_tuple(
    values: Sequence[object],
    *,
    field: str,
    cap: int,
    per_item_cap: int,
) -> tuple[str, ...]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{field} must be a list/tuple")
    if len(values) > cap:
        raise ValueError(f"{field} too many entries (cap {cap})")
    out: list[str] = []
    for v in values:
        if isinstance(v, bool) or not isinstance(v, str):
            raise ValueError(f"{field} entries must be str")
        if "\x00" in v:
            raise ValueError(f"{field} must not contain null bytes")
        if len(v) > per_item_cap:
            raise ValueError(f"{field} entry too int")
        out.append(v)
    return tuple(out)


@dataclass(frozen=True)
class CheckListTest:
    """One CheckList test definition."""

    name: str
    kind: str
    prompts: tuple[str, ...]
    expected: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, field="name"))
        object.__setattr__(self, "kind", validate_test_kind(self.kind))
        prompts = _validate_string_tuple(
            self.prompts, field="prompts",
            cap=_MAX_PROMPTS_PER_TEST, per_item_cap=_MAX_PROMPT_LEN,
        )
        if not prompts:
            raise ValueError("prompts must not be empty")
        object.__setattr__(self, "prompts", prompts)
        expected = _validate_string_tuple(
            self.expected, field="expected",
            cap=_MAX_EXPECTED_PER_TEST, per_item_cap=_MAX_EXPECTED_LEN,
        )
        # MFT + DIR require at least one expected keyword; INV does not.
        if self.kind in ("mft", "dir") and not expected:
            raise ValueError(f"{self.kind} test requires at least one expected entry")
        object.__setattr__(self, "expected", expected)


@dataclass(frozen=True)
class CheckListSpec:
    """A full CheckList suite (one or more tests, unique names)."""

    tests: tuple[CheckListTest, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.tests, tuple):
            raise ValueError("tests must be a tuple")
        if not self.tests:
            raise ValueError("tests must not be empty")
        if len(self.tests) > _MAX_TESTS:
            raise ValueError(f"too many tests (cap {_MAX_TESTS})")
        names = set()
        for t in self.tests:
            if not isinstance(t, CheckListTest):
                raise TypeError("tests must contain CheckListTest instances")
            if t.name in names:
                raise ValueError(f"duplicate test name: {t.name!r}")
            names.add(t.name)


@dataclass(frozen=True)
class CheckListTestResult:
    """Per-test pass/fail count + verdict."""

    name: str
    kind: str
    passed: int
    total: int
    verdict: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _validate_name(self.name, field="name"))
        object.__setattr__(self, "kind", validate_test_kind(self.kind))
        if isinstance(self.passed, bool) or not isinstance(self.passed, int):
            raise ValueError("passed must be int")
        if self.passed < 0:
            raise ValueError("passed must be non-negative")
        if isinstance(self.total, bool) or not isinstance(self.total, int):
            raise ValueError("total must be int")
        if self.total < 0:
            raise ValueError("total must be non-negative")
        if self.passed > self.total:
            raise ValueError("passed must not exceed total")
        if self.verdict not in _VERDICTS:
            raise ValueError(f"verdict must be one of {sorted(_VERDICTS)}")


@dataclass(frozen=True)
class CheckListReport:
    """Full CheckList report — list of results + worst-case overall."""

    results: tuple[CheckListTestResult, ...]
    overall: str

    def __post_init__(self) -> None:
        if not isinstance(self.results, tuple):
            raise ValueError("results must be a tuple")
        for r in self.results:
            if not isinstance(r, CheckListTestResult):
                raise TypeError("results must contain CheckListTestResult instances")
        if self.overall not in _VERDICTS:
            raise ValueError(f"overall must be one of {sorted(_VERDICTS)}")

    def to_dict(self) -> dict:
        return {
            "overall": self.overall,
            "results": [
                {
                    "name": r.name, "kind": r.kind,
                    "passed": r.passed, "total": r.total, "verdict": r.verdict,
                }
                for r in self.results
            ],
        }


def parse_checklist_spec(raw: object) -> CheckListSpec:
    """Parse a dict (from YAML) into a frozen ``CheckListSpec``."""
    if not isinstance(raw, dict):
        raise TypeError("checklist spec must be a dict")
    tests_raw = raw.get("tests")
    if tests_raw is None:
        raise ValueError("spec must contain a 'tests' key")
    if not isinstance(tests_raw, list):
        raise ValueError("'tests' must be a list")
    tests: list[CheckListTest] = []
    for idx, entry in enumerate(tests_raw):
        if not isinstance(entry, dict):
            raise ValueError(f"tests[{idx}] must be a dict")
        if "name" not in entry:
            raise ValueError(f"tests[{idx}] missing 'name'")
        if "kind" not in entry:
            raise ValueError(f"tests[{idx}] missing 'kind'")
        if "prompts" not in entry:
            raise ValueError(f"tests[{idx}] missing 'prompts'")
        prompts = entry["prompts"]
        if not isinstance(prompts, (list, tuple)):
            # Surface the offending test index up-front (review M4 fix —
            # was relying on `_validate_string_tuple` for a generic message).
            raise ValueError(f"tests[{idx}].prompts must be a list/tuple")
        expected = entry.get("expected") or []
        if not isinstance(expected, (list, tuple)):
            raise ValueError(f"tests[{idx}].expected must be a list/tuple")
        tests.append(CheckListTest(
            name=entry["name"],
            kind=entry["kind"],
            prompts=tuple(prompts),
            expected=tuple(expected),
        ))
    return CheckListSpec(tests=tuple(tests))


def load_checklist_spec(path: str) -> CheckListSpec:
    """Load a CheckList spec from a YAML file under cwd.

    Uses the shared :func:`enforce_under_cwd_and_no_symlink` helper for
    containment, then opens with ``O_NOFOLLOW`` (POSIX) and uses
    ``os.fstat`` on the SAME file descriptor for size enforcement
    (review H-NEW-1 fix — double-lstat-on-path is a TOCTOU race the
    attacker can win by swapping the file between calls). On Windows
    ``O_NOFOLLOW`` is absent but the OS does not follow symlinks in
    `os.open` by default and the containment check is the primary gate.
    """
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    enforce_under_cwd_and_no_symlink(path, "spec_path")
    # O_NOFOLLOW prevents the open from following a symlink planted
    # between the helper's lstat and this open() — defence-in-depth.
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):  # POSIX only
        flags |= os.O_NOFOLLOW
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        raise
    except OSError as exc:
        if isinstance(exc, FileNotFoundError):
            raise
        raise ValueError(f"cannot open path: {type(exc).__name__}") from exc
    try:
        st = os.fstat(fd)
        if stat.S_ISLNK(st.st_mode):  # impossible under O_NOFOLLOW, defence-in-depth
            raise ValueError("path must not be a symlink")
        if st.st_size > _MAX_FILE_BYTES:
            raise ValueError(
                f"spec file too large ({st.st_size} > {_MAX_FILE_BYTES})"
            )
        with os.fdopen(fd, "r", encoding="utf-8", closefd=True) as fh:
            text = fh.read()
            fd = -1  # ownership transferred to fdopen / closefd=True
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid YAML: {exc}") from exc
    return parse_checklist_spec(raw)


def _mft_pass(response: str, expected: Sequence[str]) -> bool:
    """MFT pass: any expected keyword appears as a whole WORD in the response.

    Whole-word match (review M2 fix — substring would let ``"and"`` pass on
    ``"sand"``); case-insensitive. Mirrors the v0.65.0 Part B
    ``behavior_battery._agreement_rate`` policy.
    """
    lower = response.lower()
    for kw in expected:
        target = kw.lower().strip()
        if not target:
            continue
        if re.search(rf"\b{re.escape(target)}\b", lower):
            return True
    return False


def _inv_pass(responses: Sequence[str], expected: Sequence[str]) -> bool:
    """INV pass: all responses share the same normalised content.

    If `expected` is non-empty, every response must contain at least one
    expected keyword. Otherwise, all responses must whitespace-normalise to
    the SAME non-empty value (review M-NEW-3 fix — all-whitespace responses
    used to collapse to a single empty-string key and spuriously pass INV).
    """
    if not responses:
        return False
    if expected:
        return all(_mft_pass(r, expected) for r in responses)
    normalised = {" ".join(r.lower().split()) for r in responses}
    if len(normalised) != 1:
        return False
    # Reject the empty-string degenerate case.
    return next(iter(normalised)) != ""


def _dir_pass(response: str, expected: Sequence[str]) -> bool:
    """DIR pass: any expected keyword appears."""
    return _mft_pass(response, expected)


def _classify_pass_rate(rate: float) -> str:
    """OK / MINOR / MAJOR thresholds — mirror v0.56 / v0.65 Part B."""
    if rate >= 0.85:
        return "OK"
    if rate >= 0.60:
        return "MINOR"
    return "MAJOR"


def run_checklist_spec(
    spec: CheckListSpec,
    *,
    evidence: Optional[Mapping[str, Sequence[str]]] = None,
) -> CheckListReport:
    """Run all tests in ``spec`` against operator-supplied responses.

    ``evidence`` maps each test ``name`` to the list of responses
    (one per prompt). Tests with no evidence get a neutral OK verdict.
    """
    if not isinstance(spec, CheckListSpec):
        raise TypeError("spec must be a CheckListSpec")
    if evidence is not None and not isinstance(evidence, dict):
        raise TypeError("evidence must be None or a dict")

    results: list[CheckListTestResult] = []
    overall = "OK"
    for t in spec.tests:
        if evidence is None or t.name not in evidence:
            # Neutral OK for missing evidence (matches v0.56 / v0.61 policy).
            result = CheckListTestResult(
                name=t.name, kind=t.kind, passed=0, total=0, verdict="OK",
            )
            results.append(result)
            continue
        responses = evidence[t.name]
        if not isinstance(responses, (list, tuple)):
            result = CheckListTestResult(
                name=t.name, kind=t.kind, passed=0, total=1, verdict="MAJOR",
            )
            results.append(result)
            if overall != "MAJOR":
                overall = "MAJOR"
            continue
        # Reject obviously-bad rows (non-str entries) up front.
        responses = [
            r for r in responses
            if isinstance(r, str) and not isinstance(r, bool)
        ]

        if t.kind == "mft":
            passed = sum(1 for r in responses if _mft_pass(r, t.expected))
            total = len(t.prompts)
            verdict = _classify_pass_rate(passed / total if total else 1.0)
        elif t.kind == "inv":
            # INV is a single yes/no test on the whole response set.
            if len(responses) < len(t.prompts):
                passed, total, verdict = 0, 1, "MAJOR"
            else:
                ok = _inv_pass(responses, t.expected)
                passed, total = (1 if ok else 0), 1
                verdict = "OK" if ok else "MAJOR"
        else:  # dir
            passed = sum(1 for r in responses if _dir_pass(r, t.expected))
            total = len(t.prompts)
            verdict = _classify_pass_rate(passed / total if total else 1.0)

        result = CheckListTestResult(
            name=t.name, kind=t.kind,
            passed=passed, total=total, verdict=verdict,
        )
        results.append(result)
        if verdict == "MAJOR":
            overall = "MAJOR"
        elif verdict == "MINOR" and overall == "OK":
            overall = "MINOR"

    return CheckListReport(results=tuple(results), overall=overall)
