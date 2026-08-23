"""#503 - the Transformers ``<5.0.0`` ceiling is load-bearing, not caution.

``pyproject.toml`` caps ``transformers<5.0.0`` in the ``train`` extra. The bound was
set on 2026-05-08, *before* transformers 5 existed, so it reads like a precaution --
and it was nearly lifted on exactly that reasoning. It is not a precaution.

Measured against ``transformers==5.15.1``: 45 tests fail, none of them in Soup's own
code. transformers 5 changed the return type of a private helper that ``trl`` imports::

    4.57.6:  _is_package_available("weave") -> False         # bool, falsey
    5.15.1:  _is_package_available("weave") -> (False, None) # tuple, TRUTHY

The tuple comes back for present packages too, so the inversion fires on *absent*
ones: "not installed" reads as true. ``trl/import_utils.py`` derives its optional
dependency flags from that helper, so the guarded ``import weave`` runs anyway and
``dpo_trainer`` / ``grpo_trainer`` never import behind the ``ModuleNotFoundError``.
Thirteen module-level flags in trl 0.19.1 are built this way; weave merely fires
first.

The trl boundary is exact, and it is why this ceiling cannot be lifted on its own:
trl 0.28.0 still imports the private helper, while trl 0.29.0 vendors its own
``_is_package_available`` over ``importlib.util.find_spec`` and is immune. That is
one version above Soup's own ``trl<0.29`` cap, which is itself held by an unrelated
break (``BasePairwiseJudge`` moved to ``trl.experimental.judges``). The chain is
``transformers<5`` <- ``trl<0.29`` <- the judge import; lifting one link does nothing.

This module holds two different kinds of guard, and the distinction matters:

``TestCeilingIsPinned`` is a **ratchet**. It fails on purpose when the bound moves,
because moving it without re-measuring is the exact mistake being prevented. Its
failure message names what to re-measure.

``TestDocsMatchTheDeclaredBounds`` is a **consistency check**, not a ratchet. It
asserts that the docs warn *exactly when* the two extras are provably disjoint. If
upstream is repaired and the extras become co-installable, it does not block that
work -- it tells you to delete the now-false warning. A guard that fires on correct
code gets deleted, so this one is written not to.
"""

from __future__ import annotations

import re
from pathlib import Path

from packaging.specifiers import SpecifierSet
from packaging.version import Version

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
BACKENDS_DOC = REPO_ROOT / "docs" / "backends-and-ops.md"

ISSUE_URL = "https://github.com/MakazhanAlpamys/Soup/issues/503"

# The sentence docs/backends-and-ops.md uses to state the incompatibility. Matched as
# a phrase rather than a whole paragraph so rewording the prose around it is free.
_DOCS_WARNING = "cannot be installed together"

# Probe grid for disjointness. `packaging` has no specifier-intersection primitive, so
# emptiness is decided by candidate sampling. The grid spans well past both bounds and
# includes each declared bound exactly, which is where an off-by-one would live.
_PROBE_VERSIONS = tuple(
    Version(f"{major}.{minor}.0") for major in range(3, 10) for minor in range(0, 60)
)


def _extra_body(pyproject: str, extra: str) -> str:
    """Return the raw dependency list body for one optional-dependency extra."""
    block = re.search(
        rf"^{re.escape(extra)}\s*=\s*\[(.*?)^\]",
        pyproject,
        re.MULTILINE | re.DOTALL,
    )
    assert block, f"pyproject.toml has no `{extra}` optional-dependency list"
    return block.group(1)


def _transformers_requirement(extra_body: str, extra: str) -> str:
    """Return the full transformers requirement string declared by one extra."""
    requirement = re.search(r'"(transformers(?:\s*[<>=!~][^"]*)?)"', extra_body)
    assert requirement, f"the `{extra}` extra declares no transformers requirement"
    return requirement.group(1)


def _specifier(requirement: str) -> SpecifierSet:
    return SpecifierSet(requirement[len("transformers") :].strip())


def _declared(extra: str) -> SpecifierSet:
    pyproject = PYPROJECT.read_text(encoding="utf-8")
    body = _extra_body(pyproject, extra)
    return _specifier(_transformers_requirement(body, extra))


def _extras_are_disjoint() -> bool:
    """True when no transformers version can satisfy both `train` and `mlx`."""
    train = _declared("train")
    mlx = _declared("mlx")
    probes = list(_PROBE_VERSIONS)
    # Include each declared bound verbatim: a `<=` / `<` slip lives exactly there.
    for spec in (train, mlx):
        for clause in spec:
            try:
                probes.append(Version(clause.version))
            except Exception:  # pragma: no cover - a malformed pin fails elsewhere
                continue
    return not any(
        train.contains(v, prereleases=True) and mlx.contains(v, prereleases=True)
        for v in probes
    )


class TestCeilingIsPinned:
    """Deliberate ratchet: the bound may move, but not silently."""

    def test_train_extra_still_caps_transformers_below_5(self) -> None:
        train = _declared("train")
        upper = [c for c in train if c.operator in {"<", "<="}]

        assert upper, (
            "The `train` extra no longer caps transformers. That cap is load-bearing: "
            f"transformers 5 inverts trl's optional-import guards. See {ISSUE_URL} "
            "for the measurement and the acceptance criteria before removing it."
        )
        assert [(c.operator, c.version) for c in upper] == [("<", "5.0.0")], (
            f"The transformers ceiling moved to {[str(c) for c in upper]}. Before "
            "raising it, re-measure: in a clean venv install that transformers with "
            "Soup's trl range and confirm `import trl.trainer.dpo_trainer` and "
            "`import trl.trainer.grpo_trainer` both succeed, then run the full suite "
            f"and record the result on {ISSUE_URL}. Update this test in the same PR."
        )

    def test_the_ceiling_carries_its_reason_in_pyproject(self) -> None:
        """A bare bound is what gets lifted as a precaution. Keep the why next to it."""
        pyproject = PYPROJECT.read_text(encoding="utf-8")
        assert "503" in pyproject, (
            "The comment explaining why transformers is capped below 5.0.0 lost its "
            f"reference to {ISSUE_URL}. Without the measurement attached, the bound "
            "reads like caution and gets lifted -- which is how this issue was filed."
        )


class TestDocsMatchTheDeclaredBounds:
    """Consistency, not a ratchet: this never blocks repairing the incompatibility."""

    def test_docs_warn_exactly_when_the_extras_cannot_co_install(self) -> None:
        disjoint = _extras_are_disjoint()
        warned = _DOCS_WARNING in BACKENDS_DOC.read_text(encoding="utf-8")

        if disjoint:
            assert warned, (
                "`train` and `mlx` declare transformers ranges with no version in "
                "common, so `pip install \"soup-cli[train,mlx]\"` fails with an "
                "unreadable ResolutionImpossible. docs/backends-and-ops.md must say "
                f"so in plain words (phrase: {_DOCS_WARNING!r}). See {ISSUE_URL}."
            )
        else:
            assert not warned, (
                "The `train` and `mlx` extras now share at least one transformers "
                "version, so they can co-install -- but docs/backends-and-ops.md "
                f"still says they {_DOCS_WARNING}. Delete that warning and close "
                f"{ISSUE_URL}; do not leave the docs claiming a resolved conflict."
            )

    def test_mlx_extra_requires_a_transformers_the_train_extra_forbids(self) -> None:
        """Pins the shape of the conflict, so a one-sided edit is visible."""
        mlx = _declared("mlx")
        lower = [c for c in mlx if c.operator in {">=", ">", "=="}]
        assert lower, (
            "The `mlx` extra no longer pins a transformers floor. It exists to "
            "restate mlx-lm's own `transformers>=5.0.0`; if that changed upstream, "
            f"re-check whether the conflict in {ISSUE_URL} still holds."
        )
