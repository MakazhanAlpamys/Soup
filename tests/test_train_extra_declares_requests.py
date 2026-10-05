"""The ``[train]`` extra must declare ``requests`` itself.

trl imports ``requests`` at module scope in ``trl/generation/vllm_client.py``
and ``trl.trainer.grpo_trainer`` / ``trl.experimental.online_dpo`` import that
module unconditionally, yet trl's metadata lists ``requests`` only under its
optional ``vllm`` extra. A Soup training install therefore received ``requests``
by accident, through ``datasets`` (``requests>=2.32.2``). datasets 5.1.0
(2026-10-05) stopped depending on it, and from that moment a fresh
``pip install 'soup-cli[train]'`` produced an environment in which
``task: grpo`` and ``task: online_dpo`` fail at import with
``ModuleNotFoundError: No module named 'requests'``.

These tests pin the declaration, which is the part this repository owns. The
runtime half is carried by the model-backed smoke tests (``tests/test_grpo.py``,
``tests/test_issue300_online_dpo_train.py``): they import the real trl trainers
in the environment pip resolved from this file.

pyproject is parsed with regex rather than ``tomllib`` because ``tomllib`` is
3.11+ and this repo supports 3.10 — the same convention as
``tests/test_issue636_torch_floor.py``.
"""

from __future__ import annotations

import pathlib
import re

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = pathlib.Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"

# The floor datasets declared through 5.0.1, i.e. what every training install
# resolved until the transitive route disappeared.
_FLOOR_BEFORE_THE_ROUTE_WENT_AWAY = Version("2.32.2")


def _train_extra_requirements() -> list[Requirement]:
    text = PYPROJECT.read_text(encoding="utf-8")
    table = re.search(
        r"^\[project\.optional-dependencies\]\s*$(.*?)^\[", text, re.M | re.S
    )
    assert table, "pyproject.toml has no [project.optional-dependencies] table"
    train = re.search(r"^train\s*=\s*\[(.*?)^\]", table.group(1), re.M | re.S)
    assert train, "no train = [...] array in [project.optional-dependencies]"
    # Either TOML string quote, an optional trailing comma and an optional
    # trailing comment: a marker is normally written with the other quote
    # inside ("requests; extra == 'vllm'"), so both spellings must parse.
    entries = [
        double or single
        for double, single in re.findall(
            r"""^\s*(?:"([^"]+)"|'([^']+)')\s*,?\s*(?:#.*)?$""", train.group(1), re.M
        )
    ]
    assert entries, "the [train] extra parsed to no requirement strings"
    return [Requirement(entry) for entry in entries]


def _train_requirement(name: str) -> Requirement | None:
    for requirement in _train_extra_requirements():
        if requirement.name.lower() == name:
            return requirement
    return None


class TestTrainExtraDeclaresRequests:
    def test_the_parser_sees_the_known_entries(self) -> None:
        # Guards the regex itself: a parser that silently returned one entry
        # would make the assertions below vacuous in the passing direction.
        names = {requirement.name.lower() for requirement in _train_extra_requirements()}
        assert {"torch", "transformers", "peft", "trl", "datasets", "accelerate"} <= names

    def test_requests_is_a_direct_requirement_of_the_train_extra(self) -> None:
        # Mutation check: delete the "requests>=..." line from [train] and this
        # fails by name.
        requirement = _train_requirement("requests")
        assert requirement is not None, (
            "the [train] extra does not declare requests: trl's GRPO and online-DPO "
            "trainers import it at module scope and no other [train] requirement "
            "is guaranteed to bring it (datasets stopped at 5.1.0)"
        )

    def test_the_requirement_applies_to_every_train_install(self) -> None:
        # A marker ("; extra == ...", "; sys_platform == ...") would bring the
        # failure back for the installs it excludes.
        requirement = _train_requirement("requests")
        assert requirement is not None
        assert requirement.marker is None, (
            f"requests is declared with a marker ({requirement.marker}); it must be "
            "unconditional inside [train]"
        )
        assert not requirement.extras

    def test_the_floor_is_the_one_installs_already_resolved(self) -> None:
        requirement = _train_requirement("requests")
        assert requirement is not None
        floors = [
            Version(spec.version)
            for spec in requirement.specifier
            if spec.operator in (">=", "~=", "==")
        ]
        assert floors, f"requests is declared without a floor: {requirement}"
        assert max(floors) >= _FLOOR_BEFORE_THE_ROUTE_WENT_AWAY
        # No ceiling: trl 0.29 uses only requests.Session, adapters.HTTPAdapter
        # and ConnectionError, and trl's own metadata leaves requests unbounded.
        ceilings = [spec for spec in requirement.specifier if spec.operator in ("<", "<=")]
        assert not ceilings, f"requests must not be capped in [train]: {requirement}"
