"""Shell completion script generators + dynamic value completers.

`soup completions <shell>` emits a sourceable bash / zsh / fish script.
The dynamic completers (``complete_recipe_name`` /
``complete_target_modules``) are exposed for use as
``shell_complete=...`` callbacks on Typer options.

Live config introspection (probe the operator's actual ``base`` model
for its layer names) lands in v0.64.1; v0.64.0 ships canonical Llama-
shape defaults that cover ~80% of common bases.
"""

from __future__ import annotations

from typing import List, Optional

SUPPORTED_SHELLS = frozenset({"bash", "zsh", "fish"})
_MAX_SHELL_LEN = 32

# Canonical attention/mlp module names that cover Llama / Qwen / Mistral
# / Gemma / Phi families. When ``base`` is supplied and we can probe its
# config, we'd return only what's actually there — that lookup is the
# v0.64.1 deliverable.
_DEFAULT_TARGET_MODULES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
    "embed_tokens",
)


def validate_shell(value: object) -> str:
    """Normalise + validate a shell name against ``SUPPORTED_SHELLS``."""
    if isinstance(value, bool):
        raise TypeError("shell must be str, not bool")
    if not isinstance(value, str):
        raise TypeError(f"shell must be str, got {type(value).__name__}")
    if not value:
        raise ValueError("shell must be non-empty")
    if "\x00" in value:
        raise ValueError("shell must not contain null bytes")
    if len(value) > _MAX_SHELL_LEN:
        raise ValueError(f"shell name too long (> {_MAX_SHELL_LEN} chars)")
    normalised = value.lower().strip()
    if normalised not in SUPPORTED_SHELLS:
        allowed = ", ".join(sorted(SUPPORTED_SHELLS))
        raise ValueError(f"unknown shell {value!r}; known: {allowed}")
    return normalised


def render_bash_script() -> str:
    """Render a bash completion script for the `soup` CLI.

    Defers to Typer/Click's built-in ``COMPLETE`` env machinery so the
    completion stays in sync with the live Typer app (new commands /
    flags are picked up automatically).
    """
    return (
        "# Soup bash completion (v0.64.0)\n"
        "# Source this file from ~/.bashrc:\n"
        "#   eval \"$(soup completions bash)\"\n"
        "_soup_complete() {\n"
        "    local IFS=$'\\n'\n"
        "    local response\n"
        "    response=$(env COMP_WORDS=\"${COMP_WORDS[*]}\" \\\n"
        "        COMP_CWORD=$COMP_CWORD \\\n"
        "        _SOUP_COMPLETE=bash_complete \\\n"
        "        $1 2>/dev/null)\n"
        "    for completion in $response; do\n"
        "        IFS=',' read type value <<< \"$completion\"\n"
        "        if [[ $type == 'plain' ]]; then\n"
        "            COMPREPLY+=(\"$value\")\n"
        "        fi\n"
        "    done\n"
        "    return 0\n"
        "}\n"
        "complete -o nosort -F _soup_complete soup\n"
    )


def render_zsh_script() -> str:
    """Render a zsh completion script for `soup`."""
    return (
        "#compdef soup\n"
        "# Soup zsh completion (v0.64.0)\n"
        "# Source this file from ~/.zshrc:\n"
        "#   eval \"$(soup completions zsh)\"\n"
        "_soup_complete() {\n"
        "    local -a completions\n"
        "    local -a completions_with_descriptions\n"
        "    local -a response\n"
        "    response=(\"${(@f)$(env COMP_WORDS=\"${words[*]}\" \\\n"
        "        COMP_CWORD=$((CURRENT-1)) \\\n"
        "        _SOUP_COMPLETE=zsh_complete soup 2>/dev/null)}\")\n"
        "    for type_value in \"${response[@]}\"; do\n"
        "        IFS=',' read -r -A parts <<< \"$type_value\"\n"
        "        completions+=(\"${parts[2]}\")\n"
        "    done\n"
        "    _describe '' completions\n"
        "}\n"
        "compdef _soup_complete soup\n"
    )


def render_fish_script() -> str:
    """Render a fish completion script for `soup`."""
    return (
        "# Soup fish completion (v0.64.0)\n"
        "# Source this file from ~/.config/fish/completions/soup.fish\n"
        "function _soup_complete\n"
        "    set -l response (env _SOUP_COMPLETE=fish_complete \\\n"
        "        COMP_WORDS=(commandline -cp) \\\n"
        "        COMP_CWORD=(commandline -t) soup 2>/dev/null)\n"
        "    for item in $response\n"
        "        set parts (string split \",\" $item)\n"
        "        echo $parts[2]\n"
        "    end\n"
        "end\n"
        "complete -c soup -f -a \"(_soup_complete)\"\n"
    )


def render_completion_script(shell: object) -> str:
    """Dispatch on shell name. Validates + renders one of the three scripts."""
    normalised = validate_shell(shell)
    if normalised == "bash":
        return render_bash_script()
    if normalised == "zsh":
        return render_zsh_script()
    if normalised == "fish":
        return render_fish_script()
    # Unreachable thanks to ``validate_shell``; defensive default.
    raise ValueError(f"unhandled shell {normalised!r}")


def complete_recipe_name(prefix: object) -> List[str]:
    """Suggest recipe names matching ``prefix`` (case-insensitive).

    Backed by ``soup_cli.recipes.catalog.list_recipes`` (lazy import).
    """
    if isinstance(prefix, bool):
        raise TypeError("prefix must be str, not bool")
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be str, got {type(prefix).__name__}")
    if "\x00" in prefix:
        # Defensive: shell completers should never raise.
        return []
    try:
        from soup_cli.recipes.catalog import RECIPES
    except ImportError:  # pragma: no cover
        return []
    p = prefix.lower()
    return [name for name in RECIPES if name.lower().startswith(p)]


def complete_target_modules(
    prefix: object,
    *,
    base: Optional[str] = None,
) -> List[str]:
    """Suggest ``target_modules`` values for the chosen ``base`` model.

    v0.64.0 returns the canonical Llama-shape defaults filtered by
    ``prefix``. Live per-base introspection (load the HF config, walk
    its module tree) is the v0.64.1 deliverable.
    """
    if isinstance(prefix, bool):
        raise TypeError("prefix must be str, not bool")
    if not isinstance(prefix, str):
        raise TypeError(f"prefix must be str, got {type(prefix).__name__}")
    if base is not None and not isinstance(base, str):
        raise TypeError(f"base must be str | None, got {type(base).__name__}")
    if "\x00" in prefix:
        return []
    # base-specific introspection deferred to v0.64.1; fall through to
    # the canonical defaults for now.
    return [m for m in _DEFAULT_TARGET_MODULES if m.startswith(prefix)]


__all__ = [
    "SUPPORTED_SHELLS",
    "complete_recipe_name",
    "complete_target_modules",
    "render_bash_script",
    "render_completion_script",
    "render_fish_script",
    "render_zsh_script",
    "validate_shell",
]
