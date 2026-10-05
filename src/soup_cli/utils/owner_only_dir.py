"""A folder only the account running Soup can change — created that way, re-applied on reuse.

The layer-stream primary cache lives under ``~/.soup`` (or a contained override), which the
user's profile already keeps private. A stripe root (``SOUP_LAYER_STREAM_STRIPE_DIRS``) is on
another drive by definition, and a folder there inherits that drive's ACL: on a Windows data
volume the default grants Authenticated Users *Modify*, so any local account could rewrite the
layer files a later run trusts, or read a gated base. This module gives such a folder the
primary cache's protection directly:

- **POSIX:** created with mode 0o700. On reuse the folder must belong to this user and be
  writable by nobody else (group/other write refused by name), and is set to 0o700 again. The
  mode change goes through an ``O_NOFOLLOW`` descriptor, so a link swapped in is never followed.
- **Windows:** created by ``CreateDirectoryW`` with a protected DACL — full control for this
  account and SYSTEM, inherited by everything created inside, nothing inherited from the drive
  — so no other account ever holds a handle-able window on it. On reuse its owner must be this
  account (or SYSTEM / Administrators, who can take any folder anyway), and the protected DACL
  is applied again with ``SetNamedSecurityInfoW``. That call propagates the inheritable entries
  to the files already inside, so files Soup wrote there (they inherit from the folder) are
  covered without a recursive walk; a file whose own DACL is protected is left as it is, and no
  other account could have created one in a folder that was private from its first instant.

A folder that cannot be made so raises :class:`OwnerOnlyError`; the caller refuses the run.
ctypes only: no new dependency, no subprocess. Nothing here deletes anything.
"""

import functools
import os
import stat
from typing import Any, List

#: The DACL on Windows: this account and SYSTEM, full control, inherited by files and folders.
_SYSTEM_SID = "S-1-5-18"
_ADMINISTRATORS_SID = "S-1-5-32-544"


class OwnerOnlyError(OSError):
    """A folder that cannot be made (or proven) private to the account running Soup."""


def ensure_owner_only_dir(path: str) -> bool:
    """Create ``path`` owner-only, or verify and restrict an existing one. True if created.

    The caller has already refused a link at ``path``; this still never follows one.
    """
    if os.name == "nt":
        created = _win_create_dir(path)
        if not created:
            owner = _win_owner_sid(path)
            if owner not in _win_trusted_owner_sids():
                raise OwnerOnlyError(
                    f"{path} is owned by {owner}, not by the account running Soup "
                    f"({_win_current_user_sid()})"
                )
    else:
        created = _posix_create_dir(path)
        _posix_check_existing(path)
    try:
        _restrict(path)
    except OwnerOnlyError:
        raise
    except OSError as exc:
        raise OwnerOnlyError(f"could not restrict {path} to its owner ({exc})") from exc
    return created


def _restrict(path: str) -> None:
    """Apply the owner-only permission to ``path`` (idempotent)."""
    if os.name == "nt":
        _win_protect(path)
    else:
        _posix_restrict(path)


# -- POSIX ------------------------------------------------------------------------------------
def _current_uid() -> int:
    return os.geteuid()


def _open_dir_no_follow(path: str) -> int:
    """Open ``path`` as a folder, refusing a symlink there (``O_NOFOLLOW`` applied at open).

    Goes through the shared ``open_no_follow`` helper (#820) rather than spelling the flag here,
    so the repo-wide ratchet on bare ``O_NOFOLLOW`` sites does not grow.
    """
    from soup_cli.utils.paths import open_no_follow

    return open_no_follow(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))


def _posix_create_dir(path: str) -> bool:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        return False
    return True


def _posix_check_existing(path: str) -> None:
    try:
        fd = _open_dir_no_follow(path)
    except OSError as exc:
        raise OwnerOnlyError(f"{path} cannot be opened as a real folder ({exc})") from exc
    try:
        info = os.fstat(fd)
    finally:
        os.close(fd)
    if not stat.S_ISDIR(info.st_mode):
        raise OwnerOnlyError(f"{path} is not a folder")
    uid = _current_uid()
    if info.st_uid != uid:
        raise OwnerOnlyError(f"{path} belongs to another user (uid {info.st_uid}, not {uid})")
    mode = stat.S_IMODE(info.st_mode)
    if mode & 0o022:
        raise OwnerOnlyError(f"{path} can be written by users other than its owner (mode {mode:o})")


def _posix_restrict(path: str) -> None:
    fd = _open_dir_no_follow(path)
    try:
        os.fchmod(fd, 0o700)
    finally:
        os.close(fd)


# -- Windows ----------------------------------------------------------------------------------
_SE_FILE_OBJECT = 1
_OWNER_SECURITY_INFORMATION = 0x1
_DACL_SECURITY_INFORMATION = 0x4
_PROTECTED_DACL_SECURITY_INFORMATION = 0x80000000
_SDDL_REVISION_1 = 1
_TOKEN_QUERY = 0x0008
_TOKEN_USER = 1
_TOKEN_OWNER = 4
_ERROR_ALREADY_EXISTS = 183


@functools.lru_cache(maxsize=None)
def _win() -> Any:
    """advapi32 / kernel32 with prototypes, on private handles (never the shared ``windll``)."""
    import ctypes
    from ctypes import wintypes
    from types import SimpleNamespace

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    void_p = ctypes.c_void_p
    p_void_p = ctypes.POINTER(void_p)

    kernel32.GetCurrentProcess.argtypes = []
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [void_p]
    kernel32.LocalFree.restype = void_p
    kernel32.CreateDirectoryW.argtypes = [wintypes.LPCWSTR, void_p]
    kernel32.CreateDirectoryW.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = [void_p, ctypes.POINTER(wintypes.LPWSTR)]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        p_void_p,
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = wintypes.BOOL
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.argtypes = [
        void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPWSTR),
        ctypes.POINTER(wintypes.ULONG),
    ]
    advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW.restype = wintypes.BOOL
    advapi32.GetSecurityDescriptorDacl.argtypes = [
        void_p,
        ctypes.POINTER(wintypes.BOOL),
        p_void_p,
        ctypes.POINTER(wintypes.BOOL),
    ]
    advapi32.GetSecurityDescriptorDacl.restype = wintypes.BOOL
    advapi32.GetNamedSecurityInfoW.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        p_void_p,
        p_void_p,
        p_void_p,
        p_void_p,
        p_void_p,
    ]
    advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD
    advapi32.SetNamedSecurityInfoW.argtypes = [
        wintypes.LPWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        void_p,
        void_p,
        void_p,
        void_p,
    ]
    advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD

    class _SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", void_p),
            ("bInheritHandle", wintypes.BOOL),
        ]

    return SimpleNamespace(
        ctypes=ctypes,
        wintypes=wintypes,
        advapi32=advapi32,
        kernel32=kernel32,
        SecurityAttributes=_SecurityAttributes,
    )


def _win_error(what: str) -> OSError:
    api = _win()
    code = api.ctypes.get_last_error()
    return OSError(code, f"{what} failed: {api.ctypes.FormatError(code).strip()}")


def _win_sid_string(sid: Any) -> str:
    api = _win()
    text = api.wintypes.LPWSTR()
    if not api.advapi32.ConvertSidToStringSidW(sid, api.ctypes.byref(text)):
        raise _win_error("ConvertSidToStringSidW")
    try:
        return str(text.value)
    finally:
        api.kernel32.LocalFree(api.ctypes.cast(text, api.ctypes.c_void_p))


def _win_token_sid(info_class: int) -> str:
    """The SID string in TOKEN_USER / TOKEN_OWNER of this process (both start with a PSID)."""
    api = _win()
    ctypes, wintypes = api.ctypes, api.wintypes
    token = wintypes.HANDLE()
    if not api.advapi32.OpenProcessToken(
        api.kernel32.GetCurrentProcess(), _TOKEN_QUERY, ctypes.byref(token)
    ):
        raise _win_error("OpenProcessToken")
    try:
        size = wintypes.DWORD()
        api.advapi32.GetTokenInformation(token, info_class, None, 0, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not api.advapi32.GetTokenInformation(
            token, info_class, buffer, size, ctypes.byref(size)
        ):
            raise _win_error("GetTokenInformation")
        sid = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p))[0]
        return _win_sid_string(ctypes.c_void_p(sid))
    finally:
        api.kernel32.CloseHandle(token)


def _win_current_user_sid() -> str:
    return _win_token_sid(_TOKEN_USER)


def _win_trusted_owner_sids() -> List[str]:
    """This account; its token's default owner (Administrators when elevated); SYSTEM and
    Administrators, who can take ownership of any folder on the box anyway."""
    return [_win_current_user_sid(), _win_token_sid(_TOKEN_OWNER), _SYSTEM_SID, _ADMINISTRATORS_SID]


def _win_owner_only_sddl() -> str:
    return f"D:P(A;OICI;FA;;;{_win_current_user_sid()})(A;OICI;FA;;;SY)"


class _WinDescriptor:
    """A self-relative security descriptor from SDDL, freed on exit."""

    def __init__(self, sddl: str):
        self.sddl = sddl
        self.pointer: Any = None

    def __enter__(self) -> Any:
        api = _win()
        self.pointer = api.ctypes.c_void_p()
        if not api.advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW(
            self.sddl, _SDDL_REVISION_1, api.ctypes.byref(self.pointer), None
        ):
            raise _win_error("ConvertStringSecurityDescriptorToSecurityDescriptorW")
        return self.pointer

    def __exit__(self, *_exc: Any) -> None:
        if self.pointer:
            _win().kernel32.LocalFree(self.pointer)


def _win_create_dir(path: str) -> bool:
    """CreateDirectoryW with the owner-only DACL from the first instant; False if it exists."""
    api = _win()
    with _WinDescriptor(_win_owner_only_sddl()) as descriptor:
        attributes = api.SecurityAttributes(
            api.ctypes.sizeof(api.SecurityAttributes), descriptor, False
        )
        if api.kernel32.CreateDirectoryW(path, api.ctypes.byref(attributes)):
            return True
        # Read before the descriptor is freed: LocalFree would overwrite the saved error.
        code = api.ctypes.get_last_error()
    if code == _ERROR_ALREADY_EXISTS:
        return False
    raise OSError(code, f"CreateDirectoryW({path!r}) failed: {api.ctypes.FormatError(code)}")


def _win_owner_sid(path: str) -> str:
    api = _win()
    ctypes = api.ctypes
    owner = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    code = api.advapi32.GetNamedSecurityInfoW(
        path,
        _SE_FILE_OBJECT,
        _OWNER_SECURITY_INFORMATION,
        ctypes.byref(owner),
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    if code:
        raise OSError(code, f"GetNamedSecurityInfoW({path!r}) failed: {ctypes.FormatError(code)}")
    try:
        return _win_sid_string(owner)
    finally:
        api.kernel32.LocalFree(descriptor)


def _win_protect(path: str) -> None:
    """Replace the DACL with the protected owner-only one; children inherit it."""
    api = _win()
    ctypes, wintypes = api.ctypes, api.wintypes
    with _WinDescriptor(_win_owner_only_sddl()) as descriptor:
        present, defaulted = wintypes.BOOL(), wintypes.BOOL()
        dacl = ctypes.c_void_p()
        if not api.advapi32.GetSecurityDescriptorDacl(
            descriptor, ctypes.byref(present), ctypes.byref(dacl), ctypes.byref(defaulted)
        ):
            raise _win_error("GetSecurityDescriptorDacl")
        code = api.advapi32.SetNamedSecurityInfoW(
            path,
            _SE_FILE_OBJECT,
            _DACL_SECURITY_INFORMATION | _PROTECTED_DACL_SECURITY_INFORMATION,
            None,
            None,
            dacl,
            None,
        )
    if code:
        raise OSError(code, f"SetNamedSecurityInfoW({path!r}) failed: {ctypes.FormatError(code)}")


def _win_dacl_sddl(path: str) -> str:
    """``path``'s DACL as SDDL — for tests and for a human reading a refusal."""
    api = _win()
    ctypes, wintypes = api.ctypes, api.wintypes
    descriptor = ctypes.c_void_p()
    code = api.advapi32.GetNamedSecurityInfoW(
        path,
        _SE_FILE_OBJECT,
        _DACL_SECURITY_INFORMATION,
        None,
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    if code:
        raise OSError(code, f"GetNamedSecurityInfoW({path!r}) failed: {ctypes.FormatError(code)}")
    try:
        text = wintypes.LPWSTR()
        if not api.advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW(
            descriptor, _SDDL_REVISION_1, _DACL_SECURITY_INFORMATION, ctypes.byref(text), None
        ):
            raise _win_error("ConvertSecurityDescriptorToStringSecurityDescriptorW")
        try:
            return str(text.value)
        finally:
            api.kernel32.LocalFree(ctypes.cast(text, ctypes.c_void_p))
    finally:
        api.kernel32.LocalFree(descriptor)
