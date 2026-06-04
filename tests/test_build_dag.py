"""Tests for `resolve_transform` in `soup_cli.utils.build_dag`.

Covers:
  - Built-in transform resolution.
  - Per-call ``extra`` map precedence over built-ins and dotted paths.
  - Dotted-path resolution (happy path + error cases).
  - Arity validation via ``inspect.signature``.
"""

from __future__ import annotations

import sys
import types

import pytest

# ---------------------------------------------------------------------------
# Helpers — create a temporary module on the fly for dotted-path tests.
# ---------------------------------------------------------------------------

def _make_transform_module(
    name: str,
    func_name: str,
    arity: int = 2,
    kind: str = "callable",
) -> str:
    """Write a temporary Python module and return its importable name.

    Registers the module in ``sys.modules`` so ``importlib.import_module``
    can find it.

    *kind* controls what ``func_name`` resolves to::

        callable  → a function accepting ``arity`` positional params
        non_callable → an int (non-callable)
        partial   → a function with only 1 positional param (wrong arity)
    """
    mod = types.ModuleType(name)
    if kind == "callable":
        def _fn(row, config):
            return dict(row)
        setattr(mod, func_name, _fn)
    elif kind == "non_callable":
        setattr(mod, func_name, 42)
    elif kind == "partial":
        def _fn_one(row):
            return dict(row)
        setattr(mod, func_name, _fn_one)
    elif kind == "kwonly":
        def _fn_kwonly(*, row, config):
            return dict(row)
        setattr(mod, func_name, _fn_kwonly)
    elif kind == "three_args":
        def _fn_three(row, config, extra):
            return dict(row)
        setattr(mod, func_name, _fn_three)
    mod.__file__ = f"<test_{name}>"
    sys.modules[name] = mod
    return name


# ---------------------------------------------------------------------------
# Built-in transforms
# ---------------------------------------------------------------------------


class TestResolveTransformBuiltins:
    """resolve_transform returns built-ins by name."""

    def test_identity(self):
        from soup_cli.utils.build_dag import BUILTIN_TRANSFORMS, resolve_transform

        fn = resolve_transform("identity")
        assert fn is BUILTIN_TRANSFORMS["identity"]

    def test_drop_empty(self):
        from soup_cli.utils.build_dag import BUILTIN_TRANSFORMS, resolve_transform

        fn = resolve_transform("drop_empty")
        assert fn is BUILTIN_TRANSFORMS["drop_empty"]

    def test_lowercase(self):
        from soup_cli.utils.build_dag import BUILTIN_TRANSFORMS, resolve_transform

        fn = resolve_transform("lowercase")
        assert fn is BUILTIN_TRANSFORMS["lowercase"]

    def test_add_field(self):
        from soup_cli.utils.build_dag import BUILTIN_TRANSFORMS, resolve_transform

        fn = resolve_transform("add_field")
        assert fn is BUILTIN_TRANSFORMS["add_field"]

    def test_token_count(self):
        from soup_cli.utils.build_dag import BUILTIN_TRANSFORMS, resolve_transform

        fn = resolve_transform("token_count")
        assert fn is BUILTIN_TRANSFORMS["token_count"]

    def test_unknown_raises_value_error(self):
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(ValueError, match="unknown transform"):
            resolve_transform("does_not_exist_xyz")


# ---------------------------------------------------------------------------
# Extra map precedence
# ---------------------------------------------------------------------------


class TestResolveTransformExtraPrecedence:
    """Per-call ``extra`` shadows built-ins and dotted paths."""

    def test_extra_shadows_builtin(self):
        from soup_cli.utils.build_dag import resolve_transform

        custom = lambda row, config: {"custom": True}  # noqa: E731
        fn = resolve_transform("identity", extra={"identity": custom})
        assert fn is custom

    def test_extra_shadows_dotted_path(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_shadow", "my_transform", arity=2, kind="callable"
        )
        dotted = f"{mod_name}:my_transform"

        custom = lambda row, config: {"custom": True}  # noqa: E731
        fn = resolve_transform(dotted, extra={dotted: custom})
        assert fn is custom

    def test_extra_non_callable_raises_type_error(self):
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(TypeError, match="must be callable"):
            resolve_transform("foo", extra={"foo": 42})


# ---------------------------------------------------------------------------
# Dotted-path happy path
# ---------------------------------------------------------------------------


class TestResolveTransformDottedPath:
    """Valid dotted paths are imported and returned."""

    def test_happy_path(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_hp", "my_transform", arity=2, kind="callable"
        )
        fn = resolve_transform(f"{mod_name}:my_transform")
        assert callable(fn)
        assert fn({"a": 1}, {}) == {"a": 1}

    def test_nested_module_path(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_nested", "nested_fn", arity=2, kind="callable"
        )
        fn = resolve_transform(f"{mod_name}:nested_fn")
        assert callable(fn)

    def test_lru_cache_serves_same_result(self):
        """Resolving the same dotted path twice returns the identical object."""
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_cache", "cached_fn", arity=2, kind="callable"
        )
        fn1 = resolve_transform(f"{mod_name}:cached_fn")
        fn2 = resolve_transform(f"{mod_name}:cached_fn")
        assert fn1 is fn2


# ---------------------------------------------------------------------------
# Dotted-path error cases
# ---------------------------------------------------------------------------


class TestResolveTransformDottedPathErrors:
    """Invalid dotted paths raise descriptive ValueErrors."""

    def test_unknown_module(self):
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(ValueError, match="cannot import module"):
            resolve_transform("nonexistent_module_xyz_123:something")

    def test_non_callable_attribute(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_noncall", "not_a_fn", arity=2, kind="non_callable"
        )
        with pytest.raises(ValueError, match="is not callable"):
            resolve_transform(f"{mod_name}:not_a_fn")

    def test_wrong_arity_one_arg(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_arity1", "one_arg", arity=2, kind="partial"
        )
        with pytest.raises(ValueError, match="exactly 2 positional arguments"):
            resolve_transform(f"{mod_name}:one_arg")

    def test_wrong_arity_three_args(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_arity3", "three_args", arity=2, kind="three_args"
        )
        with pytest.raises(ValueError, match="exactly 2 positional arguments"):
            resolve_transform(f"{mod_name}:three_args")

    def test_missing_attribute(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_missing", "exists_fn", arity=2, kind="callable"
        )
        with pytest.raises(ValueError, match="has no attribute"):
            resolve_transform(f"{mod_name}:does_not_exist")

    def test_empty_module_part(self):
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(ValueError, match="non-empty on both sides"):
            resolve_transform(":some_function")

    def test_empty_function_part(self):
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(ValueError, match="non-empty on both sides"):
            resolve_transform("some_module:")

    def test_no_colon_in_name_still_raises_unknown(self):
        """A name without ':' that isn't a builtin raises unknown-transform."""
        from soup_cli.utils.build_dag import resolve_transform

        with pytest.raises(ValueError, match="unknown transform"):
            resolve_transform("random_string_without_colon")


# ---------------------------------------------------------------------------
# Extra map shadows manifest dotted-path strings (integration-level)
# ---------------------------------------------------------------------------


class TestResolveTransformExtraShadowsManifest:
    """When the same name appears in ``extra`` and as a dotted path, extra wins."""

    def test_extra_overrides_dotted_path(self):
        from soup_cli.utils.build_dag import resolve_transform

        mod_name = _make_transform_module(
            "_soup_test_mod_override", "manifest_fn", arity=2, kind="callable"
        )
        dotted = f"{mod_name}:manifest_fn"

        # First call: resolves the dotted path.
        fn_from_manifest = resolve_transform(dotted)
        assert callable(fn_from_manifest)

        # Second call with extra: the extra entry shadows the dotted path.
        custom = lambda row, config: {"from_extra": True}  # noqa: E731
        fn_from_extra = resolve_transform(
            dotted, extra={dotted: custom}
        )
        assert fn_from_extra is custom

    def test_builtin_overridden_by_extra(self):
        from soup_cli.utils.build_dag import resolve_transform

        custom = lambda row, config: {"overridden": True}  # noqa: E731
        fn = resolve_transform("identity", extra={"identity": custom})
        assert fn is custom
        assert fn({"x": 1}, {}) == {"overridden": True}
