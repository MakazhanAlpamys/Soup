"""Find the external programs Soup runs, by absolute path, on PATH.

Soup starts an external program (nvidia-smi, git, docker, ssh, ...) by the
absolute path :func:`find_tool` returns, never by its bare name. Only the
absolute PATH directories are searched, in order; empty and relative entries
are skipped and the current directory is never searched.

:func:`find_tool` and :func:`tool_environment` use nothing but ``os``:
:func:`controller_source` copies their source into the controller scripts that
``soup train --cloud runpod`` and ``--cloud lambda`` write, which run without
soup_cli. Their annotations are quoted: the copy has no
``from __future__ import annotations`` and must not evaluate them.
"""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping

#: What a filtered environment for a child program always keeps: the program
#: search path and, for Windows, the executable extensions, the Windows
#: directory and the setting that keeps the current directory out of the
#: program search.
ALWAYS_KEPT_ENV_NAMES: tuple[str, ...] = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "NoDefaultCurrentDirectoryInExePath",
)


class ToolNotFoundError(FileNotFoundError):
    """An external program is in none of the absolute PATH directories."""


def find_tool(name: str, path: "str | None" = None) -> "str | None":
    """The absolute path of the program ``name`` found on PATH, or None.

    The entries of ``path`` (by default the PATH variable) are searched in
    order. An entry is stripped of surrounding whitespace and double quotes;
    empty and relative entries (``.``, ``bin``, and on Windows ``\\bin`` or
    ``C:bin``) are skipped, and the current directory is never searched. On
    Windows each PATHEXT extension is tried in turn (``.COM;.EXE;.BAT;.CMD``
    when PATHEXT is unset or names no extension), and a ``name`` that already
    ends in one is tried as it is first; elsewhere the file must be executable. ``name`` is a bare
    program name: one with a directory part raises ``ValueError``.
    """
    if not name or os.path.basename(name) != name or name in (os.curdir, os.pardir):
        raise ValueError(f"{name!r} is not a bare program name")
    if path is None:
        path = os.environ.get("PATH", os.defpath)
    windows = os.name == "nt"
    if windows:
        pathext = os.environ.get("PATHEXT", "")
        suffixes = [ext.strip() for ext in pathext.split(os.pathsep) if ext.strip()]
        if not suffixes:
            suffixes = [".COM", ".EXE", ".BAT", ".CMD"]
        if os.path.splitext(name)[1].upper() in {ext.upper() for ext in suffixes}:
            suffixes.insert(0, "")
    else:
        suffixes = [""]
    for entry in path.split(os.pathsep):
        entry = entry.strip().strip('"').strip()
        if not entry or not os.path.isabs(entry):
            continue
        if windows and not os.path.splitdrive(entry)[0]:
            continue
        for suffix in suffixes:
            candidate = os.path.join(entry, name + suffix)
            if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
                return os.path.abspath(candidate)
    return None


def tool_environment(drop=(), environ=None) -> "dict":
    """A copy of ``environ`` (by default ``os.environ``) for an external program.

    The variables named in ``drop`` are left out, their names compared without
    regard to case. On Windows ``NoDefaultCurrentDirectoryInExePath=1`` is set,
    so the program's own searches for other programs skip the current
    directory as well.
    """
    source = os.environ if environ is None else environ
    unwanted = {name.upper() for name in drop}
    if os.name == "nt":
        unwanted.add("NODEFAULTCURRENTDIRECTORYINEXEPATH")
    env = {name: value for name, value in source.items() if name.upper() not in unwanted}
    if os.name == "nt":
        env["NoDefaultCurrentDirectoryInExePath"] = "1"
    return env


def require_tool(name: str, hint: str) -> str:
    """The absolute path of the program ``name`` on PATH (see :func:`find_tool`).

    Raises :class:`ToolNotFoundError` (a ``FileNotFoundError``) naming the
    program and ``hint``, which says how to install it, when it is not found.
    """
    found = find_tool(name)
    if found is None:
        raise ToolNotFoundError(
            f"{name} was not found on PATH (only absolute PATH directories are "
            f"searched). {hint}".rstrip()
        )
    return found


def filtered_environment(
    names: Iterable[str],
    environ: Mapping[str, str] | None = None,
    *,
    case_insensitive: bool | None = None,
) -> dict[str, str]:
    """The variables of ``environ`` (by default ``os.environ``) named in
    ``names`` or in :data:`ALWAYS_KEPT_ENV_NAMES`.

    Windows variable names are case-insensitive (``SystemRoot`` is
    ``SYSTEMROOT``), so there they are matched folded to upper case, and
    exactly everywhere else; ``case_insensitive`` overrides the platform rule.
    On Windows ``NoDefaultCurrentDirectoryInExePath=1`` is set, as
    :func:`tool_environment` sets it.
    """
    source = os.environ if environ is None else environ
    if case_insensitive is None:
        case_insensitive = os.name == "nt"
    wanted = set(names) | set(ALWAYS_KEPT_ENV_NAMES)
    if case_insensitive:
        folded = {name.upper() for name in wanted}
        env = {name: value for name, value in source.items() if name.upper() in folded}
    else:
        env = {name: value for name, value in source.items() if name in wanted}
    if os.name == "nt":
        env = {name: value for name, value in env.items() if name.upper() != _NO_CWD_SEARCH}
        env["NoDefaultCurrentDirectoryInExePath"] = "1"
    return env


_NO_CWD_SEARCH = "NODEFAULTCURRENTDIRECTORYINEXEPATH"


def exclude_current_directory_from_program_search() -> None:
    """On Windows, leave the current directory out of the program search of
    this process and of the programs it starts.

    Windows looks for a program started by bare name in the current directory
    before PATH unless ``NoDefaultCurrentDirectoryInExePath`` is set in the
    environment of the process that starts it. This sets it in ``os.environ``
    when it is not set already, so it applies to the programs that Soup's own
    libraries start by bare name and is inherited by every child. Elsewhere it
    does nothing: there the current directory is searched only through an
    empty or relative PATH entry.
    """
    if os.name == "nt":
        os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")


def controller_source() -> str:
    """The source of :func:`find_tool` and :func:`tool_environment`, after the
    ``import os`` they need, for a generated script that runs without soup_cli."""
    import inspect

    return "\n\n".join(
        ["import os", inspect.getsource(find_tool), inspect.getsource(tool_environment)]
    )
