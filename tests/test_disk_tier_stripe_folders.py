"""R4 fix wave — the per-model folder inside a stripe root is Soup's own, and provably so.

A stripe root sits outside the home/cwd/tmp containment the primary cache has, by design (a
second drive is never under $HOME). What that containment bought is enforced directly instead:

- B: the folder belongs to ONE primary cache. Two primary caches of the same base (per user, per
  SOUP_LAYER_STREAM_CACHE_DIR, different dtype) sharing a stripe root never read or overwrite
  each other's layer files: the folder name carries a hash of the primary cache's realpath and
  the marker records that cache.
- C: the folder is never a symlink or junction, and resolves to where it should under its
  validated root — checked before any read, write or delete in it.
- E: the folder is owner-only — mode 0o700 on POSIX, a protected owner-only DACL on Windows —
  applied at creation and re-applied on reuse; a folder that cannot be made so refuses the run.
"""

import hashlib
import json
import os
import shutil
import stat

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

import soup_cli.utils.owner_only_dir as owner_only_dir  # noqa: E402
from soup_cli.utils.layer_shard import (  # noqa: E402
    STRIPE_MARKER_NAME,
    inspect_shard_cache,
    layer_paths,
    layer_shard_path,
    shard_checkpoint,
    stripe_dirs,
)
from soup_cli.utils.stripe_roots import (  # noqa: E402
    StripeRootError,
    stripe_folder_name,
    stripe_folder_problem,
)

WINDOWS = os.name == "nt"


def _weights(tmp_path, n_layers=3):
    torch.manual_seed(0)
    state = {}
    for idx in range(n_layers):
        pre = f"model.layers.{idx}."
        state[pre + "self_attn.q_proj.weight"] = torch.randn(8, 8)
        state[pre + "mlp.down_proj.weight"] = torch.randn(8, 16)
        state[pre + "input_layernorm.weight"] = torch.randn(8)
    state["model.embed_tokens.weight"] = torch.randn(32, 8)
    state["model.norm.weight"] = torch.randn(8)
    src = tmp_path / "weights"
    src.mkdir()
    save_file(state, str(src / "model.safetensors"))
    return str(src)


@pytest.fixture
def layout(tmp_path):
    stripe = tmp_path / "stripe"
    stripe.mkdir()
    return _weights(tmp_path), str(tmp_path / "cache" / "model"), os.path.realpath(stripe)


def _folder(out, stripe):
    return stripe_dirs(out, (stripe,))[1]


def _digests(folder):
    found = {}
    for name in sorted(os.listdir(folder)):
        with open(os.path.join(folder, name), "rb") as handle:
            found[name] = hashlib.sha256(handle.read()).hexdigest()
    return found


def _inspect(out, index, stripe):
    return inspect_shard_cache(
        out, index.dtype, index.source_fingerprint, (), "none", False, "", "",
        stripe_roots=(stripe,),
    )


def _junction(target, link):
    import _winapi

    _winapi.CreateJunction(str(target), str(link))


# --------------------------------------------------------------------------------------------
# B — one stripe folder per primary cache
# --------------------------------------------------------------------------------------------
class TestTheFolderBelongsToOnePrimaryCache:
    def test_the_folder_name_is_the_slug_plus_a_hash_of_the_primary_cache(self, tmp_path):
        first = stripe_folder_name(str(tmp_path / "a" / "model"))
        second = stripe_folder_name(str(tmp_path / "b" / "model"))
        assert first.startswith("model-") and second.startswith("model-")
        assert first != second
        assert stripe_folder_name(str(tmp_path / "a" / "model")) == first  # stable

    def test_two_primary_caches_of_one_base_never_touch_each_others_files(self, layout, tmp_path):
        """The final review's Important 2: per-user or per-project primary caches with one
        machine-wide stripe variable. B shards with another dtype; A's stripe files must be
        exactly what A wrote, and A's cache must still be reusable."""
        src, _out, stripe = layout
        a_out = str(tmp_path / "cache-a" / "model")
        b_out = str(tmp_path / "cache-b" / "model")
        a_index = shard_checkpoint(src, a_out, dtype="float32", stripe_roots=(stripe,))
        a_folder = _folder(a_out, stripe)
        before = _digests(a_folder)

        b_index = shard_checkpoint(src, b_out, dtype="bfloat16", stripe_roots=(stripe,))
        b_folder = _folder(b_out, stripe)

        assert os.path.normcase(a_folder) != os.path.normcase(b_folder)
        assert _digests(a_folder) == before
        assert _inspect(a_out, a_index, stripe)[0] is not None
        assert _inspect(b_out, b_index, stripe)[0] is not None
        said = []
        shard_checkpoint(src, a_out, dtype="float32", stripe_roots=(stripe,), notify=said.append)
        assert said == [] and _digests(a_folder) == before

    def test_the_marker_records_the_primary_cache(self, layout):
        src, out, stripe = layout
        index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        with open(os.path.join(_folder(out, stripe), STRIPE_MARKER_NAME), encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload["source_fingerprint"] == index.source_fingerprint
        assert payload["primary_cache"] == os.path.normcase(os.path.realpath(out))

    def test_a_marker_copied_from_another_cache_is_a_miss_that_leaves_that_cache_alone(
        self, layout, tmp_path
    ):
        src, _out, stripe = layout
        a_out = str(tmp_path / "cache-a" / "model")
        b_out = str(tmp_path / "cache-b" / "model")
        a_index = shard_checkpoint(src, a_out, dtype="float32", stripe_roots=(stripe,))
        shard_checkpoint(src, b_out, dtype="float32", stripe_roots=(stripe,))
        a_folder, b_folder = _folder(a_out, stripe), _folder(b_out, stripe)
        b_before = _digests(b_folder)
        shutil.copyfile(
            os.path.join(b_folder, STRIPE_MARKER_NAME), os.path.join(a_folder, STRIPE_MARKER_NAME)
        )

        index, reason = _inspect(a_out, a_index, stripe)
        assert index is None and "another cache" in reason, reason
        shard_checkpoint(src, a_out, dtype="float32", stripe_roots=(stripe,))
        assert _digests(b_folder) == b_before
        assert _inspect(a_out, a_index, stripe)[0] is not None


# --------------------------------------------------------------------------------------------
# C — no link or junction at <root>/<folder>
# --------------------------------------------------------------------------------------------
class TestNoLinkAtTheFolder:
    def _plant(self, kind, target, link):
        if kind == "symlink":
            os.symlink(str(target), str(link), target_is_directory=True)
        else:
            _junction(target, link)

    def _refused_before_any_write(self, layout, tmp_path, kind):
        src, out, stripe = layout
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        self._plant(kind, elsewhere, _folder(out, stripe))
        with pytest.raises(StripeRootError) as caught:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        message = str(caught.value)
        assert "link or junction" in message and _folder(out, stripe) in message, message
        assert os.listdir(elsewhere) == []
        assert not os.path.exists(layer_shard_path(out, 0))

    @pytest.mark.requires_symlink
    def test_a_symlinked_folder_is_refused_before_any_write(self, layout, tmp_path):
        self._refused_before_any_write(layout, tmp_path, "symlink")

    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_a_junction_folder_is_refused_before_any_write(self, layout, tmp_path):
        self._refused_before_any_write(layout, tmp_path, "junction")

    def _swap_in_a_link(self, layout, tmp_path, kind):
        """A committed striped cache whose folder is then moved away and replaced by a link
        to it: the files, the marker, everything still matches — through the link."""
        src, out, stripe = layout
        index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        folder = _folder(out, stripe)
        moved = tmp_path / "moved"
        os.replace(folder, moved)
        self._plant(kind, moved, folder)
        return src, out, stripe, index, folder

    def _a_swapped_in_link_is_never_read(self, layout, tmp_path, kind):
        src, out, stripe, index, folder = self._swap_in_a_link(layout, tmp_path, kind)
        found, reason = _inspect(out, index, stripe)
        assert found is None and "link or junction" in reason, reason
        with pytest.raises(StripeRootError, match="link or junction"):
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))

    @pytest.mark.requires_symlink
    def test_a_swapped_in_symlink_is_a_miss_and_a_refusal(self, layout, tmp_path):
        self._a_swapped_in_link_is_never_read(layout, tmp_path, "symlink")

    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_a_swapped_in_junction_is_a_miss_and_a_refusal(self, layout, tmp_path):
        self._a_swapped_in_link_is_never_read(layout, tmp_path, "junction")

    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_the_runtime_refuses_a_junction_before_reading_a_header(self, layout, tmp_path):
        from soup_cli.utils.layer_stream_runtime import _require_stripe_folders

        _src, out, stripe, index, folder = self._swap_in_a_link(layout, tmp_path, "junction")
        with pytest.raises(RuntimeError, match="link or junction") as caught:
            _require_stripe_folders(out, index)
        assert folder in str(caught.value)

    def test_the_runtime_names_a_deleted_folder_under_a_present_root(self, layout):
        """Final review Minor 9: the root is there, the folder is not."""
        from soup_cli.utils.layer_stream_runtime import _require_stripe_folders
        from soup_cli.utils.stripe_roots import STRIPE_DIRS_ENV

        src, out, stripe = layout
        index = shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        folder = _folder(out, stripe)
        shutil.rmtree(folder)
        with pytest.raises(RuntimeError, match=STRIPE_DIRS_ENV) as caught:
            _require_stripe_folders(out, index)
        assert folder in str(caught.value)

    @pytest.mark.requires_symlink
    def test_a_root_swapped_for_a_link_after_validation_is_caught_at_the_folder(self, tmp_path):
        """The folder itself is a real directory, but the validated root it was joined to now
        resolves elsewhere: comparing realpaths catches what lstat on the folder cannot."""
        real = tmp_path / "real-root"
        (real / "model-abc").mkdir(parents=True)
        swapped = tmp_path / "validated-root"
        os.symlink(str(real), str(swapped), target_is_directory=True)
        folder = os.path.join(str(swapped), "model-abc")
        problem = stripe_folder_problem(folder, str(swapped))
        assert problem is not None and "resolves to" in problem, problem

    def test_a_real_folder_under_its_root_has_no_problem(self, tmp_path):
        root = os.path.realpath(tmp_path)
        folder = os.path.join(root, "model-abc")
        assert stripe_folder_problem(folder, root) is None  # missing is the caller's call
        os.mkdir(folder)
        assert stripe_folder_problem(folder, root) is None


# --------------------------------------------------------------------------------------------
# E — owner-only
# --------------------------------------------------------------------------------------------
def _sddl_aces(sddl):
    return sddl.split("(", 1)[1].rstrip(")").split(")(") if "(" in sddl else []


def _trustees(path):
    """Who the DACL of ``path`` names: SIDs, or SDDL aliases such as SY and LA."""
    return {ace.rsplit(";", 1)[1] for ace in _sddl_aces(owner_only_dir._win_dacl_sddl(path))}


def _win_sid_api():
    """advapi32 prototypes for the SID comparison below, on a private handle (test-side only)."""
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    advapi32.ConvertStringSidToSidW.argtypes = [
        wintypes.LPCWSTR, ctypes.POINTER(ctypes.c_void_p)
    ]
    advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL
    advapi32.EqualSid.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    advapi32.EqualSid.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    return ctypes, advapi32, kernel32


def _win_same_sid(left, right):
    """True when two trustee strings name the same account, whatever their spelling.

    The SDDL a DACL is rendered to writes well-known SIDs as aliases (SYSTEM as ``SY``, the
    RID-500 built-in Administrator as ``LA``), so ``S-1-5-21-...-500`` and ``LA`` are one
    trustee that no string comparison can match. ``ConvertStringSidToSidW`` accepts both
    spellings and ``EqualSid`` compares the binary SIDs."""
    ctypes, advapi32, kernel32 = _win_sid_api()
    sids = []
    try:
        for text in (left, right):
            pointer = ctypes.c_void_p()
            if not advapi32.ConvertStringSidToSidW(text, ctypes.byref(pointer)):
                raise OSError(ctypes.get_last_error(), f"not a SID or SDDL alias: {text!r}")
            sids.append(pointer)
        return bool(advapi32.EqualSid(sids[0], sids[1]))
    finally:
        for pointer in sids:
            kernel32.LocalFree(pointer)


def _assert_only_owner_and_system(trustees, where):
    """Every trustee is the current user or SYSTEM, and the current user is among them."""
    me = owner_only_dir._win_current_user_sid()
    strangers = sorted(
        t for t in trustees if not (_win_same_sid(t, me) or _win_same_sid(t, "SY"))
    )
    assert not strangers, (strangers, where)
    assert any(_win_same_sid(t, me) for t in trustees), (sorted(trustees), where)


@pytest.mark.skipif(WINDOWS, reason="POSIX mode bits")
class TestOwnerOnlyOnPosix:
    def test_a_new_folder_is_mode_0700(self, layout):
        src, out, stripe = layout
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert stat.S_IMODE(os.stat(_folder(out, stripe)).st_mode) == 0o700

    def test_a_reused_folder_is_tightened_again(self, layout):
        src, out, stripe = layout
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        folder = _folder(out, stripe)
        os.chmod(folder, 0o755)
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert stat.S_IMODE(os.stat(folder).st_mode) == 0o700

    @pytest.mark.parametrize("mode", [0o770, 0o707, 0o777])
    def test_a_folder_others_can_write_is_refused_by_name(self, layout, mode):
        src, out, stripe = layout
        folder = _folder(out, stripe)
        os.mkdir(folder)
        os.chmod(folder, mode)
        with pytest.raises(StripeRootError, match="written by users other than its owner") as e:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert folder in str(e.value)
        assert not os.path.exists(layer_shard_path(out, 0))

    def test_a_folder_owned_by_another_user_is_refused_by_name(self, layout, monkeypatch):
        src, out, stripe = layout
        folder = _folder(out, stripe)
        os.mkdir(folder, 0o700)
        monkeypatch.setattr(owner_only_dir, "_current_uid", lambda: os.stat(folder).st_uid + 1)
        with pytest.raises(StripeRootError, match="belongs to another user") as caught:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert folder in str(caught.value)


@pytest.mark.skipif(not WINDOWS, reason="Windows ACLs")
class TestOwnerOnlyOnWindows:
    def test_the_sid_comparison_reads_aliases_and_rejects_strangers(self):
        """The helper the DACL assertions stand on: a SID and its alias are one trustee."""
        assert _win_same_sid("SY", "S-1-5-18")
        assert _win_same_sid("S-1-5-32-544", "BA")  # the built-in Administrators group
        assert _win_same_sid("S-1-5-18", "S-1-5-18")
        assert not _win_same_sid("SY", "BA")
        assert not _win_same_sid("SY", owner_only_dir._win_current_user_sid())

    def test_the_rid_500_account_is_read_through_its_la_alias(self, monkeypatch):
        """A CI account that is the built-in Administrator (RID 500) has its own SID rendered
        as ``LA`` by the SDDL a DACL is read back as; that is still the current user."""
        me = owner_only_dir._win_current_user_sid()
        rid_500 = me.rsplit("-", 1)[0] + "-500"
        if not _win_same_sid("LA", rid_500):
            pytest.skip("LA is not this account's domain RID-500 here (a domain-joined box)")
        monkeypatch.setattr(owner_only_dir, "_win_current_user_sid", lambda: rid_500)
        _assert_only_owner_and_system({"LA", "SY"}, "RID 500 as LA")

    def test_an_alias_form_of_the_current_user_passes_and_a_stranger_does_not(self, monkeypatch):
        monkeypatch.setattr(owner_only_dir, "_win_current_user_sid", lambda: "S-1-5-32-544")
        _assert_only_owner_and_system({"BA", "SY"}, "alias form")
        with pytest.raises(AssertionError):
            _assert_only_owner_and_system({"BA", "SY", "AU"}, "a stranger")
        with pytest.raises(AssertionError):
            _assert_only_owner_and_system({"SY"}, "the owner missing")

    def _assert_owner_only(self, path):
        sddl = owner_only_dir._win_dacl_sddl(path)
        assert sddl.startswith("D:P"), sddl  # protected: nothing inherited from the drive
        _assert_only_owner_and_system(_trustees(path), sddl)
        assert all(ace.startswith("A;") for ace in _sddl_aces(sddl)), sddl

    def test_a_new_folder_carries_a_protected_owner_only_dacl(self, layout):
        src, out, stripe = layout
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        folder = _folder(out, stripe)
        self._assert_owner_only(folder)
        # Files Soup writes there inherit it: nothing but the owner and SYSTEM.
        shard = layer_paths(out, _idx(out))[1]
        _assert_only_owner_and_system(_trustees(shard), shard)

    def test_a_reused_folder_and_what_it_holds_are_restricted_again(self, layout):
        """A folder made by plain mkdir inherits the root's ACL (on a data drive that grants
        Authenticated Users Modify). Reuse re-applies the protected DACL, and it reaches the
        files already inside: SetNamedSecurityInfoW propagates inheritable ACEs to children."""
        src, out, stripe = layout
        folder = _folder(out, stripe)
        os.mkdir(folder)
        child = os.path.join(folder, "already-here.txt")
        with open(child, "w", encoding="utf-8") as handle:
            handle.write("x")
        assert not owner_only_dir._win_dacl_sddl(folder).startswith("D:P")
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        self._assert_owner_only(folder)
        _assert_only_owner_and_system(_trustees(child), child)

    def test_a_folder_owned_by_another_account_is_refused_by_name(self, layout, monkeypatch):
        src, out, stripe = layout
        folder = _folder(out, stripe)
        os.mkdir(folder)
        monkeypatch.setattr(owner_only_dir, "_win_owner_sid", lambda _path: "S-1-5-21-1-2-3-1001")
        with pytest.raises(StripeRootError, match="owned by S-1-5-21-1-2-3-1001") as caught:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert folder in str(caught.value)
        assert not os.path.exists(layer_shard_path(out, 0))


def _idx(out):
    from soup_cli.utils.layer_shard import read_shard_index

    return read_shard_index(out)


class TestAFolderThatCannotBeMadePrivateRefusesTheRun:
    def test_a_failed_restriction_names_the_folder_and_writes_nothing(self, layout, monkeypatch):
        src, out, stripe = layout

        def refuse(path):
            raise OSError(5, "Access is denied")

        monkeypatch.setattr(owner_only_dir, "_restrict", refuse)
        with pytest.raises(StripeRootError, match="cannot be made private") as caught:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert _folder(out, stripe) in str(caught.value)
        assert "Access is denied" in str(caught.value)
        assert not os.path.exists(layer_shard_path(out, 0))

    def test_a_reused_cache_is_restricted_again_before_it_is_trusted(self, layout, monkeypatch):
        """Reuse is the path that trusts files already on the drive, so it is the one that
        must not skip the check."""
        src, out, stripe = layout
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        calls = []
        real = owner_only_dir.ensure_owner_only_dir
        monkeypatch.setattr(
            owner_only_dir, "ensure_owner_only_dir", lambda path: calls.append(path) or real(path)
        )
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        assert [os.path.normcase(p) for p in calls] == [os.path.normcase(_folder(out, stripe))]


# --------------------------------------------------------------------------------------------
# #1614 — the root is re-checked before EVERY write of a shard, not once at secure time
# --------------------------------------------------------------------------------------------
def _plant_link(kind, target, link):
    if kind == "symlink":
        os.symlink(str(target), str(link), target_is_directory=True)
    else:
        _junction(target, link)


_LINK_KINDS = [
    pytest.param("symlink", marks=pytest.mark.requires_symlink),
    pytest.param(
        "junction",
        marks=pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point"),
    ),
]


class TestAStripeRootSwapIsRefusedAtTheNextStripeWrite:
    """``secure_stripe_folder`` runs once, before the per-layer write loop starts. Between
    then and the last write, the root could be renamed and a link substituted for it — the
    per-model folder's own lstat still looks like a real directory, so only comparing its
    realpath against the validated root catches the swap. The fix re-runs that comparison
    (``stripe_folder_problem``) before every per-layer write and before the marker write.

    5 layers over (primary, stripe): layers 1 and 3 go to the stripe root, so a swap planted
    right after one of those two writes is picked up either by the next per-layer write (the
    other one) or, when the swap happens after the last layer, by the marker write.
    """

    @pytest.mark.parametrize("kind", _LINK_KINDS)
    @pytest.mark.parametrize(
        "swap_after, refused_at", [(1, "decoder layer 3"), (3, "stripe marker")]
    )
    def test_the_swap_is_refused(self, tmp_path, monkeypatch, kind, swap_after, refused_at):
        from soup_cli.utils import layer_shard as layer_shard_module

        src = _weights(tmp_path, n_layers=5)
        out = str(tmp_path / "cache" / "model")
        (tmp_path / "stripe").mkdir()
        stripe = os.path.realpath(str(tmp_path / "stripe"))
        folder_name = os.path.basename(_folder(out, stripe))
        # Already holds a same-named folder, so the substituted link looks legitimate to
        # anything that does not compare realpaths against the originally validated root.
        elsewhere = tmp_path / "elsewhere"
        (elsewhere / folder_name).mkdir(parents=True)
        trigger = os.path.basename(layer_shard_path(out, swap_after))
        real_atomic_save = layer_shard_module._atomic_save
        swapped = []

        def fake_atomic_save(blob, path):
            real_atomic_save(blob, path)
            if (
                not swapped
                and os.path.basename(path) == trigger
                and os.path.basename(os.path.dirname(path)) == folder_name
            ):
                os.replace(stripe, str(tmp_path / "moved-root"))
                _plant_link(kind, elsewhere, stripe)
                swapped.append(path)

        monkeypatch.setattr(layer_shard_module, "_atomic_save", fake_atomic_save)
        with pytest.raises(StripeRootError) as caught:
            shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        message = str(caught.value)
        assert swapped, "the fixture never planted the link"
        assert stripe in message and refused_at in message, message
        # Nothing landed through the substituted link, and no index was committed (first shard).
        assert os.listdir(str(elsewhere / folder_name)) == []
        assert not os.path.exists(os.path.join(out, "index.json"))


# --------------------------------------------------------------------------------------------
# #1614 — a directory junction is refused as a stripe root, a volume mount point is not
# --------------------------------------------------------------------------------------------
class TestJunctionVersusVolumeMountPoint:
    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_a_directory_junction_target_is_not_a_volume_mount_point(self, tmp_path):
        from soup_cli.utils.stripe_roots import is_volume_mount_point

        target = tmp_path / "target"
        target.mkdir()
        link = tmp_path / "link"
        _junction(target, link)
        assert is_volume_mount_point(str(link)) is False

    @pytest.mark.parametrize(
        "target, expected",
        [
            (r"\\?\Volume{1be1e677-9257-45ec-936f-712a49d3af54}\\", True),
            (r"\\?\Volume{1be1e677-9257-45ec-936f-712a49d3af54}", True),
            (r"\??\Volume{1be1e677-9257-45ec-936f-712a49d3af54}\\", True),
            (r"\??\Volume{1be1e677-9257-45ec-936f-712a49d3af54}", True),
            (r"\\?\Volume{1be1e677-9257-45ec-936f-712a49d3af54}\Users\nanda", False),
            (r"\\?\C:\Users\nanda", False),
            (r"\\?\UNC\server\share\folder", False),
            (r"\\?\Volume{1be1e677-9257-45ec-936f-712a49d3af54}\data\{cache}", False),
            (r"\\?\Volume{1be1e677-9257-45ec-936f-712a49d3af54}x", False),
        ],
        ids=[
            "volume-root-trailing-sep-devicepath",
            "volume-root-no-trailing-sep",
            "volume-root-trailing-sep-ntpath",
            "volume-root-no-trailing-sep-ntpath",
            "folder-below-volume-path",
            "drive-letter-folder",
            "unc-folder",
            "folder-below-volume-path-name-ends-with-a-brace",
            "text-after-the-volume-name",
        ],
    )
    def test_the_classifier_distinguishes_a_volume_root_from_a_folder_beneath_one(
        self, tmp_path, monkeypatch, target, expected
    ):
        r"""No elevation is available to mount a real volume, so the link-target resolution is
        patched directly -- the thing the classifier actually branches on. A junction can
        target a folder beneath a volume GUID path exactly as it can beneath a drive letter or
        a UNC share (reproduced for real with ``mklink /J`` against
        ``\\?\Volume{...}\Users\...``), so the rule has to be "names the whole volume", not
        "starts with Volume{"."""
        from soup_cli.utils import stripe_roots as stripe_roots_module

        link = tmp_path / "reparse-point"
        link.mkdir()
        monkeypatch.setattr(os, "readlink", lambda path: target)
        assert stripe_roots_module.is_volume_mount_point(str(link)) is expected

    def test_a_reparse_point_that_is_not_a_link_is_not_a_volume_mount_point(
        self, tmp_path, monkeypatch
    ):
        """On Windows ``os.readlink`` raises ``ValueError``, not ``OSError``, for a reparse
        point that is neither a symlink nor a junction/mount point, so the classifier has to
        expect both -- only catching ``OSError`` let such a folder through as "not a mount
        point, so must be a junction", when ``validate_stripe_root`` should refuse it by name
        instead of letting a bare ``ValueError`` escape."""
        from soup_cli.utils import stripe_roots as stripe_roots_module

        def not_a_link(path):
            raise ValueError("not a symbolic link")

        monkeypatch.setattr(os, "readlink", not_a_link)
        assert stripe_roots_module.is_volume_mount_point(str(tmp_path)) is False

    def test_a_non_reparse_folder_is_not_a_volume_mount_point(self, tmp_path):
        from soup_cli.utils.stripe_roots import is_volume_mount_point

        assert is_volume_mount_point(str(tmp_path)) is False

    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_a_junction_stripe_root_is_refused(self, tmp_path):
        from soup_cli.utils.stripe_roots import validate_stripe_root

        target = tmp_path / "elsewhere"
        target.mkdir()
        link = tmp_path / "stripe-link"
        _junction(target, link)
        primary = tmp_path / "primary"
        primary.mkdir()
        with pytest.raises(StripeRootError, match="junction") as caught:
            validate_stripe_root(str(link), primary_root=str(primary))
        assert "stripe-link" in str(caught.value)

    @pytest.mark.skipif(not WINDOWS, reason="junctions are a Windows reparse point")
    def test_a_volume_mount_point_stripe_root_is_accepted(self, tmp_path, monkeypatch):
        """Same reparse tag as a junction; the classifier is what tells them apart, so this
        patches the classifier itself rather than the unavailable (needs elevation) real
        volume-mount-point setup."""
        from soup_cli.utils import stripe_roots as stripe_roots_module

        root = tmp_path / "mounted-volume"
        target = tmp_path / "elsewhere"
        target.mkdir()
        _junction(target, root)
        monkeypatch.setattr(stripe_roots_module, "is_volume_mount_point", lambda path: True)
        primary = tmp_path / "primary"
        primary.mkdir()
        resolved = stripe_roots_module.validate_stripe_root(
            str(root), primary_root=str(primary)
        )
        assert os.path.normcase(resolved) == os.path.normcase(os.path.realpath(str(root)))
