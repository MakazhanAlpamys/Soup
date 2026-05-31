"""Ollama integration utilities — detect, deploy, list, remove models."""

import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Ollama TEMPLATE blocks for common chat formats
OLLAMA_TEMPLATES: Dict[str, str] = {
    "chatml": (
        '{{ if .System }}<|im_start|>system\n'
        '{{ .System }}<|im_end|>\n'
        '{{ end }}{{ range .Messages }}'
        '{{ if eq .Role "user" }}<|im_start|>user\n'
        '{{ .Content }}<|im_end|>\n'
        '{{ else if eq .Role "assistant" }}<|im_start|>assistant\n'
        '{{ .Content }}<|im_end|>\n'
        '{{ end }}{{ end }}<|im_start|>assistant\n'
    ),
    "llama": (
        '<|begin_of_text|>'
        '{{ if .System }}<|start_header_id|>system<|end_header_id|>\n\n'
        '{{ .System }}<|eot_id|>{{ end }}'
        '{{ range .Messages }}'
        '{{ if eq .Role "user" }}<|start_header_id|>user<|end_header_id|>\n\n'
        '{{ .Content }}<|eot_id|>'
        '{{ else if eq .Role "assistant" }}<|start_header_id|>assistant<|end_header_id|>\n\n'
        '{{ .Content }}<|eot_id|>'
        '{{ end }}{{ end }}'
        '<|start_header_id|>assistant<|end_header_id|>\n\n'
    ),
    "mistral": (
        '{{ if .System }}[INST] {{ .System }} [/INST]\n{{ end }}'
        '{{ range .Messages }}'
        '{{ if eq .Role "user" }}[INST] {{ .Content }} [/INST]\n'
        '{{ else if eq .Role "assistant" }}{{ .Content }}</s>\n'
        '{{ end }}{{ end }}'
    ),
    "vicuna": (
        '{{ if .System }}{{ .System }}\n\n{{ end }}'
        '{{ range .Messages }}'
        '{{ if eq .Role "user" }}USER: {{ .Content }}\n'
        '{{ else if eq .Role "assistant" }}ASSISTANT: {{ .Content }}</s>\n'
        '{{ end }}{{ end }}ASSISTANT:'
    ),
    "zephyr": (
        '{{ if .System }}<|system|>\n{{ .System }}</s>\n{{ end }}'
        '{{ range .Messages }}'
        '{{ if eq .Role "user" }}<|user|>\n{{ .Content }}</s>\n'
        '{{ else if eq .Role "assistant" }}<|assistant|>\n{{ .Content }}</s>\n'
        '{{ end }}{{ end }}<|assistant|>\n'
    ),
}

# Map Soup data formats to Ollama template names
FORMAT_TO_TEMPLATE: Dict[str, str] = {
    "chatml": "chatml",
    "alpaca": "llama",
    "llama": "llama",
    "mistral": "mistral",
    "vicuna": "vicuna",
    "zephyr": "zephyr",
    "sharegpt": "chatml",
}

# Valid model name pattern: alphanumeric, hyphens, underscores, colons (for tags)
_MODEL_NAME_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9._:-]*$")

# Soup-deployed model prefix
SOUP_MODEL_PREFIX = "soup-"

# Allowed Ollama PARAMETER keys (prevents injection of arbitrary directives)
ALLOWED_OLLAMA_PARAMS = frozenset({
    "temperature", "top_p", "top_k", "num_ctx", "num_predict",
    "stop", "repeat_penalty", "repeat_last_n", "seed", "mirostat",
    "mirostat_tau", "mirostat_eta", "num_gpu", "num_thread",
    "tfs_z", "typical_p", "penalize_newline",
})


def validate_model_name(name: str) -> Tuple[bool, str]:
    """Validate Ollama model name.

    Returns (is_valid, error_message).
    """
    if not name:
        return False, "Model name cannot be empty"
    if len(name) > 128:
        return False, "Model name too long (max 128 characters)"
    # Block path separators and null bytes
    if "/" in name or "\\" in name or "\0" in name:
        return False, "Model name must not contain path separators or null bytes"
    if not _MODEL_NAME_RE.match(name):
        return False, (
            "Model name must start with alphanumeric and contain only "
            "alphanumeric, hyphens, underscores, dots, or colons"
        )
    return True, ""


def validate_gguf_path(gguf_path: Path) -> Tuple[bool, str]:
    """Validate GGUF file path — must exist and stay under cwd.

    Returns (is_valid, error_message).
    """
    if not gguf_path.exists():
        return False, f"GGUF file not found: {gguf_path}"
    if not gguf_path.is_file():
        return False, f"GGUF path is not a file: {gguf_path}"
    if gguf_path.suffix.lower() != ".gguf":
        return False, "File must have a .gguf extension"
    # Path traversal protection: resolve and check it's under cwd
    resolved = gguf_path.resolve()
    cwd = Path.cwd().resolve()
    try:
        resolved.relative_to(cwd)
    except ValueError:
        return False, "GGUF path must be under the current working directory"
    return True, ""


def detect_ollama() -> Optional[str]:
    """Check if Ollama is installed, return version string or None."""
    try:
        result = subprocess.run(
            ["ollama", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode == 0:
            output = result.stdout.strip() or result.stderr.strip()
            # Extract version from output like "ollama version is 0.6.2"
            match = re.search(r"(\d+\.\d+\.\d+)", output)
            if match:
                return match.group(1)
            return output if output else "unknown"
        return None
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None


def infer_chat_template(config_format: Optional[str]) -> Optional[str]:
    """Map a Soup config format to an Ollama template name.

    Returns template name (key into OLLAMA_TEMPLATES) or None if unknown.
    """
    if not config_format:
        return None
    return FORMAT_TO_TEMPLATE.get(config_format.lower())


def create_modelfile(
    gguf_path: Path,
    template: Optional[str] = None,
    system_prompt: Optional[str] = None,
    parameters: Optional[Dict[str, str]] = None,
) -> str:
    """Generate Ollama Modelfile content.

    Args:
        gguf_path: Path to the GGUF model file.
        template: Template name (key in OLLAMA_TEMPLATES) or raw template string.
        system_prompt: Optional system prompt.
        parameters: Optional dict of PARAMETER key=value pairs.

    Returns:
        Modelfile content as string.
    """
    lines = [f"FROM {gguf_path}"]

    # Template
    if template:
        template_content = OLLAMA_TEMPLATES.get(template, template)
        lines.append(f'TEMPLATE """{template_content}"""')

    # System prompt
    if system_prompt:
        # Escape double quotes and strip newlines in system prompt
        escaped = system_prompt.replace('"', '\\"')
        escaped = escaped.replace("\n", " ").replace("\r", "")
        lines.append(f'SYSTEM "{escaped}"')

    # Parameters — validated against allowlist, no control chars
    if parameters:
        for key, value in sorted(parameters.items()):
            if key not in ALLOWED_OLLAMA_PARAMS:
                raise ValueError(
                    f"Unknown Ollama parameter: {key!r}. "
                    f"Allowed: {', '.join(sorted(ALLOWED_OLLAMA_PARAMS))}"
                )
            if "\n" in value or "\r" in value or "\0" in value:
                raise ValueError(
                    f"Parameter value for {key!r} contains "
                    "illegal characters (newline or null)"
                )
            lines.append(f"PARAMETER {key} {value}")

    return "\n\n".join(lines) + "\n"


def deploy_to_ollama(name: str, modelfile_content: str) -> Tuple[bool, str]:
    """Deploy a model to Ollama via `ollama create`.

    Args:
        name: Model name for Ollama.
        modelfile_content: Full Modelfile content.

    Returns:
        (success, message) tuple.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".Modelfile", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(modelfile_content)
        tmp_path = tmp.name

    try:
        result = subprocess.run(
            ["ollama", "create", name, "-f", tmp_path],
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or "Model created successfully"
        return False, result.stderr.strip() or "ollama create failed"
    except FileNotFoundError:
        return False, "Ollama not found. Install from https://ollama.com"
    except subprocess.TimeoutExpired:
        return False, "ollama create timed out (5 minutes)"
    except OSError as exc:
        return False, f"Failed to run ollama: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def list_soup_models() -> List[Dict[str, str]]:
    """List Ollama models deployed by Soup (prefixed with soup-).

    Returns list of dicts with 'name' and 'size' keys.
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return []

    models = []
    for line in result.stdout.strip().splitlines()[1:]:  # Skip header
        parts = line.split()
        if not parts:
            continue
        model_name = parts[0]
        # Filter to soup- prefixed models
        if model_name.startswith(SOUP_MODEL_PREFIX) or (
            ":" in model_name and model_name.split(":")[0].startswith(SOUP_MODEL_PREFIX)
        ):
            size = parts[2] + " " + parts[3] if len(parts) >= 4 else "unknown"
            models.append({"name": model_name, "size": size})
    return models


def remove_model(name: str) -> Tuple[bool, str]:
    """Remove a model from Ollama.

    Returns (success, message) tuple.
    """
    try:
        result = subprocess.run(
            ["ollama", "rm", name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            return True, result.stdout.strip() or f"Model '{name}' removed"
        return False, result.stderr.strip() or f"Failed to remove '{name}'"
    except FileNotFoundError:
        return False, "Ollama not found. Install from https://ollama.com"
    except subprocess.TimeoutExpired:
        return False, "ollama rm timed out"
    except OSError as exc:
        return False, f"Failed to run ollama: {exc}"
