"""R4 — SOUP_LAYER_STREAM_STRIPE_DIRS: every entry validated, every refusal by name."""

import os
import subprocess
import sys

import pytest

from soup_cli.utils.paths import quote_path
from soup_cli.utils.stripe_roots import (
    MAX_STRIPE_DIRS,
    STRIPE_DIRS_ENV,
    StripeRootError,
    layer_roots_for,
    parse_stripe_dirs,
    resolve_stripe_roots,
    validate_stripe_root,
)

SEP = os.pathsep


def _env(*entries):
    return {STRIPE_DIRS_ENV: SEP.join(entries)}


@pytest.fixture
def layout(tmp_path):
    primary = tmp_path / "cache"
    primary.mkdir()
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    return str(primary), str(stripe)


class TestParsing:
    def test_unset_and_blank_mean_no_stripe(self):
        assert parse_stripe_dirs(None) == ()
        assert parse_stripe_dirs("") == ()
        assert parse_stripe_dirs(f" {SEP} ") == ()

    def test_splits_on_pathsep_and_drops_blank_entries(self):
        assert parse_stripe_dirs(f"/a{SEP}{SEP} /b ") == ("/a", "/b")


class TestOneEntry:
    def test_a_valid_folder_comes_back_as_its_realpath(self, layout):
        primary, stripe = layout
        assert validate_stripe_root(stripe, primary_root=primary) == os.path.realpath(stripe)

    def test_control_characters_are_refused(self, layout):
        primary, stripe = layout
        with pytest.raises(StripeRootError, match="control characters"):
            validate_stripe_root(stripe + "\x07", primary_root=primary)

    def test_a_relative_path_is_refused(self, layout):
        primary, _ = layout
        with pytest.raises(StripeRootError, match="absolute path"):
            validate_stripe_root("relative-dir", primary_root=primary)

    def test_a_missing_folder_is_refused_as_not_mounted(self, layout, tmp_path):
        primary, _ = layout
        with pytest.raises(StripeRootError, match="existing directory"):
            validate_stripe_root(str(tmp_path / "gone"), primary_root=primary)

    def test_an_existing_file_is_refused_as_not_a_directory(self, layout, tmp_path):
        primary, _ = layout
        entry = tmp_path / "not-a-folder.txt"
        entry.write_text("x")
        with pytest.raises(StripeRootError, match="existing directory"):
            validate_stripe_root(str(entry), primary_root=primary)

    def test_a_missing_folder_names_both_ways_out(self, layout, tmp_path):
        """Final review Recommendation 5: reconnect it, or drop it and re-shard to one root."""
        primary, _ = layout
        with pytest.raises(StripeRootError, match="unset") as caught:
            validate_stripe_root(str(tmp_path / "gone"), primary_root=primary)
        assert "Reconnect" in str(caught.value)

    @pytest.mark.parametrize("char", ["\x7f", "\x85", "\x9b"], ids=["DEL", "NEL", "CSI"])
    def test_del_and_c1_control_characters_are_refused(self, layout, char):
        """Security LOW-6: C0 alone left DEL and C1 (0x9B is a one-byte CSI) through."""
        primary, stripe = layout
        with pytest.raises(StripeRootError, match="control characters"):
            validate_stripe_root(stripe + char, primary_root=primary)

    @pytest.mark.skipif(os.name != "nt", reason="Windows drive-relative spelling")
    def test_a_path_without_a_drive_is_refused_on_windows(self, layout):
        """Security LOW-6: os.path.isabs on a path that starts with a separator but names no
        drive is True on 3.10-3.12, and the recorded root would then depend on the drive of
        the current directory."""
        primary, stripe = layout
        no_drive = os.path.splitdrive(os.path.realpath(stripe))[1]
        with pytest.raises(StripeRootError, match="must name a drive letter or a UNC share"):
            validate_stripe_root(no_drive, primary_root=primary)

    @pytest.mark.skipif(os.name != "nt", reason="Windows device-namespace paths")
    @pytest.mark.parametrize("prefix", ["\\\\?\\", "\\\\.\\"], ids=["question", "dot"])
    def test_a_device_namespace_spelling_is_refused(self, layout, prefix):
        """Security LOW-1: realpath keeps a device-namespace prefix and the overlap rule then
        compares it with the primary as a different drive, so a folder INSIDE the primary
        passed."""
        primary, _ = layout
        inner = os.path.join(primary, "inner")
        os.mkdir(inner)
        with pytest.raises(StripeRootError, match="device-namespace path is not accepted"):
            validate_stripe_root(prefix + os.path.realpath(inner), primary_root=primary)

    def test_a_folder_inside_the_primary_root_is_refused(self, layout):
        primary, _ = layout
        inner = os.path.join(primary, "inner")
        os.mkdir(inner)
        with pytest.raises(StripeRootError, match="overlaps the primary"):
            validate_stripe_root(inner, primary_root=primary)

    def test_the_same_folder_twice_is_refused(self, layout):
        primary, stripe = layout
        first = validate_stripe_root(stripe, primary_root=primary)
        with pytest.raises(StripeRootError, match="overlaps another stripe root"):
            validate_stripe_root(stripe, primary_root=primary, accepted=[first])

    def test_a_home_relative_entry_is_expanded_not_refused(self, layout, monkeypatch, tmp_path):
        """Review Focus 2: users write the primary variable with ~, and will write this one so."""
        primary, stripe = layout
        monkeypatch.setenv("HOME", str(tmp_path))
        monkeypatch.setenv("USERPROFILE", str(tmp_path))
        got = validate_stripe_root(os.path.join("~", "stripe"), primary_root=primary)
        assert got == os.path.realpath(stripe)

    @pytest.mark.requires_symlink
    def test_a_symlink_is_refused(self, layout, tmp_path):
        primary, stripe = layout
        link = tmp_path / "link"
        os.symlink(stripe, link, target_is_directory=True)
        with pytest.raises(StripeRootError, match="symlink"):
            validate_stripe_root(str(link), primary_root=primary)


class TestTheWholeVariable:
    def test_unset_is_no_stripe_and_probes_nothing(self, tmp_path):
        probed = []
        assert resolve_stripe_roots(str(tmp_path), disk_kind=probed.append, environ={}) == ()
        assert probed == []

    def test_the_same_volume_as_the_primary_is_refused(self, layout):
        primary, stripe = layout
        with pytest.raises(StripeRootError, match="same volume"):
            resolve_stripe_roots(primary, disk_kind=lambda path: "nvme", environ=_env(stripe))

    def test_a_non_nvme_volume_is_refused_naming_the_override(self, layout, monkeypatch):
        import soup_cli.utils.stripe_roots as mod

        primary, stripe = layout
        ids = {os.path.realpath(primary): 1, os.path.realpath(stripe): 2}
        monkeypatch.setattr(mod, "volume_of", lambda path: ids[os.path.realpath(path)])
        with pytest.raises(StripeRootError, match="stream_disk_kind"):
            resolve_stripe_roots(primary, disk_kind=lambda path: "ssd", environ=_env(stripe))

    def test_two_valid_drives_come_back_in_order(self, tmp_path, monkeypatch):
        import soup_cli.utils.stripe_roots as mod

        folders = [tmp_path / name for name in ("cache", "d", "e")]
        for folder in folders:
            folder.mkdir()
        ids = {os.path.realpath(folder): n for n, folder in enumerate(folders)}
        monkeypatch.setattr(mod, "volume_of", lambda path: ids[os.path.realpath(path)])
        got = resolve_stripe_roots(
            str(folders[0]),
            disk_kind=lambda path: "nvme",
            environ=_env(str(folders[1]), str(folders[2])),
        )
        assert got == (os.path.realpath(folders[1]), os.path.realpath(folders[2]))

    def test_too_many_entries_are_refused_before_any_probe(self, tmp_path):
        entries = [str(tmp_path / f"s{n}") for n in range(MAX_STRIPE_DIRS + 1)]
        probed = []
        with pytest.raises(StripeRootError, match=f"at most {MAX_STRIPE_DIRS}"):
            resolve_stripe_roots(str(tmp_path), disk_kind=probed.append, environ=_env(*entries))
        assert probed == []


class TestPlacement:
    def test_one_root_is_the_empty_placement(self):
        assert layer_roots_for(5, 1) == ()

    def test_layers_alternate_and_an_odd_count_is_fine(self):
        assert layer_roots_for(5, 2) == (0, 1, 0, 1, 0)
        assert layer_roots_for(4, 3) == (0, 1, 2, 0)

    def test_zero_roots_is_a_programming_error(self):
        with pytest.raises(ValueError, match="at least 1"):
            layer_roots_for(3, 0)


class TestQuotePath:
    """#1611 — a path in a refusal message reads as it was typed."""

    @pytest.mark.parametrize("typed", [
        "D:\\folder\\sub",            # a drive-letter path
        "\\\\server\\share\\dir",     # a UNC path
        "my dir",                     # a space survives, quoted
        "trailing\\",                 # a trailing separator
        "/posix/path",                # a POSIX path comes out unchanged
        "relative\\path",             # a relative path with backslashes
    ])
    def test_shown_as_typed_inside_single_quotes(self, typed):
        assert quote_path(typed) == f"'{typed}'"

    @pytest.mark.parametrize("code", [0x00, 0x07, 0x1B, 0x7F, 0x9B, 0x9F])
    def test_control_characters_are_escaped_not_raw(self, code):
        assert quote_path(f"a{chr(code)}b") == f"'a\\x{code:02x}b'"


class TestRefusalSpellings:
    """#1611 — every refusal names the entry as typed, not in repr's doubled form."""

    def test_a_backslash_entry_is_not_doubled(self, layout):
        primary, _ = layout
        with pytest.raises(StripeRootError) as err:
            validate_stripe_root("folder\\sub", primary_root=primary)
        message = str(err.value)
        assert "'folder\\sub'" in message
        assert "'folder\\\\sub'" not in message

    @pytest.mark.parametrize("typed", ["folder\\sub", "\\\\server\\share\\dir"])
    def test_the_relative_path_refusal_names_the_entry_as_typed(self, layout, typed):
        primary, _ = layout
        with pytest.raises(StripeRootError) as err:
            validate_stripe_root(typed, primary_root=primary)
        assert f"'{typed}'" in str(err.value)

    @pytest.mark.parametrize("code", [0x1B, 0x7F, 0x9B])
    def test_control_characters_are_shown_as_escapes(self, layout, code):
        primary, _ = layout
        entry = f"stripe{chr(code)}"
        with pytest.raises(StripeRootError) as err:
            validate_stripe_root(entry, primary_root=primary)
        message = str(err.value)
        assert chr(code) not in message, "a raw control character reached the message"
        assert f"\\x{code:02x}" in message, "the message does not show which character it was"

    @pytest.mark.skipif(os.name != "nt", reason="needs a drive-letter spelling")
    def test_a_missing_drive_letter_folder_is_shown_as_typed(self, tmp_path):
        primary = tmp_path / "cache"
        primary.mkdir()
        with pytest.raises(StripeRootError) as err:
            validate_stripe_root("D:\\does-not-exist-xyz", primary_root=str(primary))
        message = str(err.value)
        assert "'D:\\does-not-exist-xyz'" in message
        assert "'D:\\\\does-not-exist-xyz'" not in message

    @pytest.mark.skipif(os.name != "nt", reason="needs two folders on one drive")
    def test_the_same_volume_refusal_names_the_entry_as_typed(self, layout):
        primary, stripe = layout
        with pytest.raises(StripeRootError) as err:
            resolve_stripe_roots(primary, disk_kind=lambda path: "nvme", environ=_env(stripe))
        message = str(err.value)
        assert f"'{stripe}'" in message
        assert repr(stripe) not in message

    def test_the_nvme_refusal_names_the_entry_as_typed(self, layout, monkeypatch):
        import soup_cli.utils.stripe_roots as mod

        primary, stripe = layout
        ids = {os.path.realpath(primary): 1, os.path.realpath(stripe): 2}
        monkeypatch.setattr(mod, "volume_of", lambda path: ids[os.path.realpath(path)])
        with pytest.raises(StripeRootError) as err:
            resolve_stripe_roots(
                primary, disk_kind=lambda path: "ssd", environ=_env(stripe),
            )
        # A POSIX path has no spelling for repr to change, so as-typed is all
        # there is to pin here; the doubled form is covered on Windows above.
        assert f"'{stripe}'" in str(err.value)


def test_the_other_message_sites_use_quote_path():
    """#1611 — the Windows API refusals and the disk pre-flight quote paths too."""
    import pathlib

    import soup_cli

    root = pathlib.Path(soup_cli.__file__).parent
    owner = (root / "utils" / "owner_only_dir.py").read_text(encoding="utf-8")
    setup = (root / "trainer" / "stream_setup.py").read_text(encoding="utf-8")
    roots = (root / "utils" / "stripe_roots.py").read_text(encoding="utf-8")
    assert "path!r" not in owner, "owner_only_dir still renders a path with repr"
    assert "path!r" not in setup, "stream_setup still renders a path with repr"
    assert "entry!r" not in roots and "path!r" not in roots, (
        "stripe_roots still renders an entry or path with repr"
    )
    assert owner.count("quote_path(path)") == 4
    assert "quote_path(path)" in setup


def test_the_module_does_not_import_torch():
    code = "import sys, soup_cli.utils.stripe_roots; sys.exit(int('torch' in sys.modules))"
    assert subprocess.run([sys.executable, "-c", code], check=False).returncode == 0
