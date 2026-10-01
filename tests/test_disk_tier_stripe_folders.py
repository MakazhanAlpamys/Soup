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
    """Who the DACL of ``path`` names: SIDs, or SDDL aliases such as SY."""
    return {ace.rsplit(";", 1)[1] for ace in _sddl_aces(owner_only_dir._win_dacl_sddl(path))}


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
    def _assert_owner_only(self, path):
        sddl = owner_only_dir._win_dacl_sddl(path)
        assert sddl.startswith("D:P"), sddl  # protected: nothing inherited from the drive
        allowed = {owner_only_dir._win_current_user_sid(), "SY"}
        trustees = {ace.rsplit(";", 1)[1] for ace in _sddl_aces(sddl)}
        assert trustees <= allowed | {"S-1-5-18"}, sddl
        assert owner_only_dir._win_current_user_sid() in trustees, sddl
        assert all(ace.startswith("A;") for ace in _sddl_aces(sddl)), sddl

    def test_a_new_folder_carries_a_protected_owner_only_dacl(self, layout):
        src, out, stripe = layout
        shard_checkpoint(src, out, dtype="float32", stripe_roots=(stripe,))
        folder = _folder(out, stripe)
        self._assert_owner_only(folder)
        # Files Soup writes there inherit it: nothing but the owner and SYSTEM.
        trustees = _trustees(layer_paths(out, _idx(out))[1])
        assert trustees <= {owner_only_dir._win_current_user_sid(), "SY", "S-1-5-18"}, trustees

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
        trustees = _trustees(child)
        assert trustees <= {owner_only_dir._win_current_user_sid(), "SY", "S-1-5-18"}, trustees

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
