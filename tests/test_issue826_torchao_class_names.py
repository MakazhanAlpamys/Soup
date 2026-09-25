"""#826 — every torchao class Soup asks for must exist in the installed torchao.

Three of the four export schemes and the whole ``training.nvfp4`` path resolved
names torchao does not export (``NVFP4Config``, ``Int8DynActInt4Config``,
``Float8DynActFloat8Config``), so a Blackwell user who set ``nvfp4: true`` got a
yellow line and a bf16 run, and three of four ``soup export --format torchao``
schemes could not build a config at all. The refusal said "upgrade torchao"
against 0.18.0, the newest release.

The reason it survived: no test used the real package. ``test_v07121.py``
injected a fake ``torchao.quantization`` defining ``NVFP4Config``, and
``test_v0531_142.py`` used a ``MagicMock``, where ``hasattr`` is true for every
name. So the checks here import the REAL torchao and skip when it is absent;
the ``torchao-contract`` CI job installs ``.[qat]`` so they actually run.
"""

from __future__ import annotations

import importlib
import re
from pathlib import Path

import pytest

from soup_cli.utils.torchao_compat import (
    TORCHAO_CLASSES,
    TORCHAO_MIN_VERSION,
    resolve_torchao_class,
)

ROOT = Path(__file__).resolve().parents[1]


class TestTheFloorIsWrittenOnce:
    def test_pyproject_pins_the_same_floor(self):
        """Four floors disagreed (0.4.0 in [qat] and doctor, 0.5.0 in
        fp8_attention, 0.7.0 in apply_nvfp4, 0.5 in soup export), which is how a
        wrong name kept telling users to upgrade."""
        text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
        assert f'qat = ["torchao>={TORCHAO_MIN_VERSION}"]' in text, text[-800:]

    def test_no_source_file_states_its_own_torchao_floor(self):
        """A second floor in a message is the drift this replaces."""
        # Matches "torchao>=0.5.0", "torchao >= 0.5", "torchao (>=0.5.0)" and
        # "torchao[...]>=x.y": the parenthesised spelling is the one my first
        # version of this guard missed, found by mutating fp8.py's message back.
        pattern = re.compile(r"torchao[^\r\n]{0,12}?[>=]=\s*[0-9]+\.[0-9]+")
        offenders = []
        for path in sorted((ROOT / "src").rglob("*.py")):
            if path.name == "torchao_compat.py":
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if pattern.search(line):
                    offenders.append(f"{path.relative_to(ROOT).as_posix()}: {line.strip()}")
        assert offenders == [], offenders


class TestAgainstTheInstalledTorchao:
    """Skipped without torchao; the ``torchao-contract`` CI job installs it."""

    @pytest.fixture(autouse=True)
    def _require_torchao(self):
        pytest.importorskip("torchao", reason="install .[qat] to run the contract")

    @pytest.mark.parametrize("key", sorted(TORCHAO_CLASSES))
    def test_every_class_soup_asks_for_resolves(self, key):
        resolved = resolve_torchao_class(key)
        assert callable(resolved), (key, resolved)

    def test_the_names_are_the_ones_torchao_defines(self):
        """Resolution by import path, not by ``hasattr`` on the package root:
        at least one candidate per key has to define the attribute where the
        map says it does."""
        for key, candidates in TORCHAO_CLASSES.items():
            found = []
            for module_path, attribute in candidates:
                try:
                    module = importlib.import_module(module_path)
                except ImportError:
                    continue
                if hasattr(module, attribute):
                    found.append(f"{module_path}.{attribute}")
            assert found, f"{key}: none of {candidates} defines its attribute"

    def test_the_public_candidate_is_tried_before_the_prototype_one(self):
        """``torchao.prototype`` is not a promised location, so a release that
        graduates one of these must be picked up without a code change. Same
        shape as ``_trl_compat.resolve_trl_symbol`` trying ``trl`` before
        ``trl.experimental``."""
        for key, candidates in TORCHAO_CLASSES.items():
            paths = [module_path for module_path, _ in candidates]
            prototypes = [i for i, p in enumerate(paths) if p.startswith("torchao.prototype")]
            publics = [i for i, p in enumerate(paths) if not p.startswith("torchao.prototype")]
            if prototypes and publics:
                assert max(publics) < min(prototypes), (key, paths)

    def test_nvfp4_training_uses_the_training_config_not_an_inference_one(self):
        """``quantize_(model, <inference config>)`` is post-training weight
        quantization. Only the training config replaces ``nn.Linear`` with
        ``NVFP4Linear`` and quantises the backward GEMMs too, which is what
        ``training.nvfp4`` claims."""
        attributes = {attribute for _, attribute in TORCHAO_CLASSES["NVFP4Training"]}
        assert attributes == {"NVFP4TrainingConfig"}
        doc = resolve_torchao_class("NVFP4Training").__doc__ or ""
        assert "backward" in doc.lower(), doc[:200]

    def test_int4_takes_group_size_and_not_inner_k_tiles(self):
        """The export allowlist advertised ``inner_k_tiles``; 0.18.0 raises
        TypeError for it, so the key promised a knob that cannot be passed."""
        factory = resolve_torchao_class("Int4WeightOnly")
        assert factory(group_size=64) is not None
        with pytest.raises(TypeError, match="inner_k_tiles"):
            factory(inner_k_tiles=8)

    def test_the_export_allowlist_only_offers_keys_the_class_accepts(self):
        """Every advertised kwarg must be a real field of the class it reaches.

        Reads the shipped allowlist rather than keeping a copy: the first version
        of this test hardcoded the same dict, so re-adding ``inner_k_tiles``
        (which 0.18.0 rejects with TypeError) left it green."""
        import dataclasses

        from soup_cli.utils.save_formats import TORCHAO_SCHEME_KWARGS

        for scheme, keys in TORCHAO_SCHEME_KWARGS.items():
            factory = resolve_torchao_class(scheme)
            fields = {field.name for field in dataclasses.fields(factory)}
            assert set(keys) <= fields, (scheme, set(keys) - fields)


class TestTheOrderOfTheChecks:
    def test_absent_torchao_is_absent_even_with_its_submodules_cached(self, monkeypatch):
        """The root package is imported before any submodule.

        ``import_module("torchao.prototype....")`` answers from ``sys.modules``
        when anything earlier in the process imported it, so with only the root
        set to None the resolver sailed past a torchao that is not importable
        and the "torchao is missing" branch was unreachable. Found by the
        maintainer on #1066: ``test_v07121.py`` passed alone and failed when
        this file ran first. So this test deliberately leaves the submodules
        cached — deleting them is what hid the bug.
        """
        import sys

        pytest.importorskip("torchao", reason="install .[qat] to run the contract")
        resolve_torchao_class("NVFP4Training")  # warm the submodule cache
        assert any(name.startswith("torchao.") for name in sys.modules)

        monkeypatch.setitem(sys.modules, "torchao", None)
        with pytest.raises(RuntimeError, match="Could not import torchao"):
            resolve_torchao_class("NVFP4Training")


    def test_a_rejected_kwarg_is_a_kwarg_error_even_without_torchao(self, monkeypatch):
        """The allowlist runs before the class is resolved. Resolving first turned
        "that key is not allowed" into "torchao is missing", which sends the user
        to fix the wrong thing."""
        import sys

        from soup_cli.utils.save_formats import export_torchao

        # Every cached torchao submodule has to go, not just the root: an earlier
        # test in this file imports them, and importlib then resolves
        # torchao.prototype... from the cache even with the root set to None,
        # which made this test pass in file order and fail alone.
        for name in [n for n in sys.modules if n == "torchao" or n.startswith("torchao.")]:
            monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setitem(sys.modules, "torchao", None)
        monkeypatch.chdir(ROOT)
        with pytest.raises(ValueError, match="not allowed"):
            export_torchao(
                model_dir="src",
                output_dir="src",
                scheme="NVFP4",
                quant_config_data={"group_size": 32},
            )


class TestTheResolverLoop:
    """The loop itself, with stub modules, so these run in the 3x3 matrix where
    torchao is absent.

    Both tests exist because the maintainer mutated ``resolve_torchao_class`` on
    #1066 and the suite stayed green. On the real 0.18.0 every candidate imports
    and the first hit is the answer, so neither the ORDER the loop walks nor the
    ``continue`` on a failed import is exercised by anything the installed
    package can show.
    """

    @staticmethod
    def _install(monkeypatch, modules):
        """Put a fake ``torchao`` and the given submodules in ``sys.modules``.

        A value of ``None`` is how the import system spells "this import fails":
        ``import_module`` raises ``ImportError`` on it, which is the branch a
        future torchao that drops a prototype submodule would take.
        """
        import sys
        import types

        monkeypatch.setitem(sys.modules, "torchao", types.ModuleType("torchao"))
        for name, value in modules.items():
            monkeypatch.setitem(sys.modules, name, value)

    def test_the_public_candidate_wins_when_both_modules_define_the_name(
        self, monkeypatch
    ):
        """``test_the_public_candidate_is_tried_before_the_prototype_one`` checks
        the order of the ``TORCHAO_CLASSES`` tuple, not that the resolver follows
        it — reversing the loop leaves it green. Here the two modules define the
        attribute as *different* objects, so only the returned identity can say
        which was tried first.

        This is the case that matters when a release graduates a prototype class:
        both paths answer for a while, and Soup has to take the promised one.
        """
        import types

        public_module = types.ModuleType("torchao.quantization")
        prototype_module = types.ModuleType("torchao.prototype.stub")

        class PublicConfig:
            pass

        class PrototypeConfig:
            pass

        public_module.TheConfig = PublicConfig
        prototype_module.TheConfig = PrototypeConfig
        self._install(
            monkeypatch,
            {
                "torchao.quantization": public_module,
                "torchao.prototype.stub": prototype_module,
            },
        )
        monkeypatch.setitem(
            TORCHAO_CLASSES,
            "StubKey",
            (
                ("torchao.quantization", "TheConfig"),
                ("torchao.prototype.stub", "TheConfig"),
            ),
        )

        assert resolve_torchao_class("StubKey") is PublicConfig
        assert PublicConfig is not PrototypeConfig, "sanity: distinguishable"

    def test_a_candidate_that_fails_to_import_is_skipped_not_fatal(
        self, monkeypatch
    ):
        """A failed candidate import must ``continue`` to the next one. Replacing
        that ``continue`` with a raise or a break is green against 0.18.0, where
        every candidate imports; it is the path a torchao that removed a
        prototype submodule would take, and it would turn a working fallback into
        a hard failure on upgrade.
        """
        import types

        survivor = types.ModuleType("torchao.prototype.survivor")

        class Survivor:
            pass

        survivor.Survivor = Survivor
        self._install(
            monkeypatch,
            {
                "torchao.quantization": None,  # ImportError on import_module
                "torchao.prototype.survivor": survivor,
            },
        )
        monkeypatch.setitem(
            TORCHAO_CLASSES,
            "StubKey",
            (
                ("torchao.quantization", "Survivor"),
                ("torchao.prototype.survivor", "Survivor"),
            ),
        )

        assert resolve_torchao_class("StubKey") is Survivor

    def test_an_absent_root_is_fatal_even_with_a_candidate_cached(
        self, monkeypatch
    ):
        """The stub twin of
        ``test_absent_torchao_is_absent_even_with_its_submodules_cached``, which
        needs the real package and therefore skips on the 3x3 matrix — where
        torchao is absent and this branch is the whole point.

        Dropping the root import leaves both green *there*: the cached submodule
        answers and the resolver never notices torchao itself cannot be imported.
        """
        import types

        cached = types.ModuleType("torchao.prototype.cached")

        class Cached:
            pass

        cached.Cached = Cached
        self._install(monkeypatch, {"torchao.prototype.cached": cached})
        import sys

        monkeypatch.setitem(sys.modules, "torchao", None)
        monkeypatch.setitem(
            TORCHAO_CLASSES, "StubKey", (("torchao.prototype.cached", "Cached"),)
        )

        with pytest.raises(RuntimeError, match="Could not import torchao"):
            resolve_torchao_class("StubKey")

    def test_the_message_names_every_candidate_when_none_answers(
        self, monkeypatch
    ):
        """Control for the two above: when the loop really does run out, the error
        carries both paths — so a passing fallback above cannot be a resolver that
        silently returns something on any input."""
        import types

        empty = types.ModuleType("torchao.prototype.empty")
        self._install(
            monkeypatch,
            {
                "torchao.quantization": None,
                "torchao.prototype.empty": empty,
            },
        )
        monkeypatch.setitem(
            TORCHAO_CLASSES,
            "StubKey",
            (
                ("torchao.quantization", "Missing"),
                ("torchao.prototype.empty", "Missing"),
            ),
        )

        with pytest.raises(RuntimeError) as excinfo:
            resolve_torchao_class("StubKey")
        message = str(excinfo.value)
        assert "torchao.quantization.Missing" in message
        assert "torchao.prototype.empty.Missing (not defined there)" in message


class TestTheExportSchemesAreCovered:
    def test_every_shipped_scheme_has_a_resolution(self):
        """A scheme the CLI advertises but the map cannot resolve is the defect."""
        from soup_cli.utils.save_formats import TORCHAO_PTQ_SCHEMES

        missing = [s for s in sorted(TORCHAO_PTQ_SCHEMES) if s not in TORCHAO_CLASSES]
        assert missing == [], missing
