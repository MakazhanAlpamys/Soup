"""v0.53.0 Parts A+B — UD GGUF + IQ / Apple-ARM quant schema helpers.

Schema-only support for Unsloth Dynamic 2.0 GGUF ladder (``UD-Q8_K_XL`` …
``UD-IQ1_M``), the IQ1/IQ2/IQ3 family, the Apple/ARM-friendly Q4_NL /
Q5.x / Q4.x variants, and the existing TQ1_0 1.58-bit GGUF flavour (from
v0.52.0 Part D, re-exposed here for ``soup export --format gguf-iq``).

Live llama.cpp ``imatrix`` calibration + actual GGUF write are deferred
to v0.53.1 (mirrors v0.50.0 stub-then-live pattern).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Optional, cast

# --- Part A: Unsloth Dynamic 2.0 GGUF ladder ---------------------------------
UD_GGUF_FORMATS: frozenset[str] = frozenset({
    "UD-Q8_K_XL",
    "UD-Q6_K_XL",
    "UD-Q5_K_XL",
    "UD-Q4_K_XL",
    "UD-Q3_K_XL",
    "UD-Q2_K_XL",
    "UD-IQ4_XS",
    "UD-IQ3_M",
    "UD-IQ3_XXS",
    "UD-IQ2_M",
    "UD-IQ2_XS",
    "UD-IQ2_XXS",
    "UD-IQ1_M",
    "UD-IQ1_S",
})

# --- Part B: IQ + Apple/ARM quant flavours -----------------------------------
IQ_GGUF_FORMATS: frozenset[str] = frozenset({
    "IQ1_S",
    "IQ1_M",
    "IQ2_XXS",
    "IQ2_XS",
    "IQ2_S",
    "IQ2_M",
    "IQ3_XXS",
    "IQ3_XS",
    "IQ3_S",
    "IQ3_M",
    "IQ4_XS",
    "IQ4_NL",
})
APPLE_ARM_GGUF_FORMATS: frozenset[str] = frozenset({
    "Q4_0_4_4",
    "Q4_0_4_8",
    "Q4_0_8_8",
    "Q4_NL",
    "Q5_0",
    "Q5_1",
    "Q5_K_S",
    "Q5_K_M",
    "Q4_K_S",
    "Q4_K_M",
})

# Union of all v0.53.0 schema-only GGUF flavours (excludes TQ1_0 which is owned
# by v0.52.0 Part D ``utils/bitnet.py``; ``is_advanced_gguf_format`` returns
# True for the BitNet family too via the helper below for export-CLI parity).
ALL_ADVANCED_GGUF_FORMATS: frozenset[str] = (
    UD_GGUF_FORMATS | IQ_GGUF_FORMATS | APPLE_ARM_GGUF_FORMATS
)

_MAX_FORMAT_LEN: int = 32


@dataclass(frozen=True)
class GGUFQuantSpec:
    """Frozen metadata for a v0.53.0 advanced GGUF flavour."""

    name: str
    family: str          # "ud" | "iq" | "apple_arm"
    bits: float
    description: str
    live_wired: bool


def _spec(name: str, family: str, bits: float, description: str) -> GGUFQuantSpec:
    return GGUFQuantSpec(
        name=name, family=family, bits=bits,
        description=description, live_wired=False,
    )


_GGUF_METADATA: Mapping[str, GGUFQuantSpec] = MappingProxyType({
    # Unsloth Dynamic 2.0 ladder
    "UD-Q8_K_XL": _spec("UD-Q8_K_XL", "ud", 8.0, "UD Q8_K_XL (Unsloth Dynamic 2.0)"),
    "UD-Q6_K_XL": _spec("UD-Q6_K_XL", "ud", 6.0, "UD Q6_K_XL"),
    "UD-Q5_K_XL": _spec("UD-Q5_K_XL", "ud", 5.0, "UD Q5_K_XL"),
    "UD-Q4_K_XL": _spec("UD-Q4_K_XL", "ud", 4.0, "UD Q4_K_XL"),
    "UD-Q3_K_XL": _spec("UD-Q3_K_XL", "ud", 3.0, "UD Q3_K_XL"),
    "UD-Q2_K_XL": _spec("UD-Q2_K_XL", "ud", 2.0, "UD Q2_K_XL"),
    "UD-IQ4_XS":  _spec("UD-IQ4_XS",  "ud", 4.0, "UD IQ4_XS"),
    "UD-IQ3_M":   _spec("UD-IQ3_M",   "ud", 3.0, "UD IQ3_M"),
    "UD-IQ3_XXS": _spec("UD-IQ3_XXS", "ud", 3.0, "UD IQ3_XXS"),
    "UD-IQ2_M":   _spec("UD-IQ2_M",   "ud", 2.0, "UD IQ2_M"),
    "UD-IQ2_XS":  _spec("UD-IQ2_XS",  "ud", 2.0, "UD IQ2_XS"),
    "UD-IQ2_XXS": _spec("UD-IQ2_XXS", "ud", 2.0, "UD IQ2_XXS"),
    "UD-IQ1_M":   _spec("UD-IQ1_M",   "ud", 1.0, "UD IQ1_M (smallest UD)"),
    "UD-IQ1_S":   _spec("UD-IQ1_S",   "ud", 1.0, "UD IQ1_S"),
    # IQ family (non-UD)
    "IQ1_S":   _spec("IQ1_S",   "iq", 1.0, "IQ1_S 1-bit"),
    "IQ1_M":   _spec("IQ1_M",   "iq", 1.0, "IQ1_M 1-bit"),
    "IQ2_XXS": _spec("IQ2_XXS", "iq", 2.0, "IQ2_XXS 2-bit"),
    "IQ2_XS":  _spec("IQ2_XS",  "iq", 2.0, "IQ2_XS 2-bit"),
    "IQ2_S":   _spec("IQ2_S",   "iq", 2.0, "IQ2_S 2-bit"),
    "IQ2_M":   _spec("IQ2_M",   "iq", 2.0, "IQ2_M 2-bit"),
    "IQ3_XXS": _spec("IQ3_XXS", "iq", 3.0, "IQ3_XXS 3-bit"),
    "IQ3_XS":  _spec("IQ3_XS",  "iq", 3.0, "IQ3_XS 3-bit"),
    "IQ3_S":   _spec("IQ3_S",   "iq", 3.0, "IQ3_S 3-bit"),
    "IQ3_M":   _spec("IQ3_M",   "iq", 3.0, "IQ3_M 3-bit"),
    "IQ4_XS":  _spec("IQ4_XS",  "iq", 4.0, "IQ4_XS 4-bit"),
    "IQ4_NL":  _spec("IQ4_NL",  "iq", 4.0, "IQ4_NL 4-bit (non-linear)"),
    # Apple/ARM neural-engine-friendly
    "Q4_0_4_4": _spec("Q4_0_4_4", "apple_arm", 4.0, "Apple/ARM Q4_0_4_4"),
    "Q4_0_4_8": _spec("Q4_0_4_8", "apple_arm", 4.0, "Apple/ARM Q4_0_4_8"),
    "Q4_0_8_8": _spec("Q4_0_8_8", "apple_arm", 4.0, "Apple/ARM Q4_0_8_8"),
    "Q4_NL":    _spec("Q4_NL",    "apple_arm", 4.0, "Apple/ARM Q4_NL"),
    "Q5_0":     _spec("Q5_0",     "apple_arm", 5.0, "Apple/ARM Q5_0"),
    "Q5_1":     _spec("Q5_1",     "apple_arm", 5.0, "Apple/ARM Q5_1"),
    "Q5_K_S":   _spec("Q5_K_S",   "apple_arm", 5.0, "Apple/ARM Q5_K_S"),
    "Q5_K_M":   _spec("Q5_K_M",   "apple_arm", 5.0, "Apple/ARM Q5_K_M"),
    "Q4_K_S":   _spec("Q4_K_S",   "apple_arm", 4.0, "Apple/ARM Q4_K_S"),
    "Q4_K_M":   _spec("Q4_K_M",   "apple_arm", 4.0, "Apple/ARM Q4_K_M"),
})


def _basic_validate(value: object, field: str) -> str:
    if isinstance(value, bool):
        raise TypeError(f"{field} must not be bool, got {value!r}")
    if not isinstance(value, str):
        raise TypeError(f"{field} must be str, got {type(value).__name__}")
    if not value:
        raise ValueError(f"{field} must be non-empty")
    if "\x00" in value:
        raise ValueError(f"{field} must not contain null bytes")
    if len(value) > _MAX_FORMAT_LEN:
        raise ValueError(f"{field} too long (max {_MAX_FORMAT_LEN} chars)")
    return value


# Lowercase index built once at module load — O(1) lookup vs O(N) walk
# (code-review MEDIUM fix). Mirrors v0.32.0 ``pick_mixed_precision`` quirk
# ordering policy where sorting / indexing is precomputed.
_LOWER_INDEX: Mapping[str, str] = MappingProxyType({
    name.lower(): name for name in ALL_ADVANCED_GGUF_FORMATS
})


def _resolve_canonical(value: str) -> str | None:
    """Match ``value`` against the union allowlist case-insensitively.

    Returns the canonical (original-case) entry from the allowlist or
    ``None`` if no match.
    """
    return _LOWER_INDEX.get(value.lower())


def validate_ud_gguf_format(value: object) -> str:
    """Validate ``value`` is a UD GGUF format string. Returns canonical form."""
    _basic_validate(value, "ud_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in UD_GGUF_FORMATS:
        supported = ", ".join(sorted(UD_GGUF_FORMATS))
        raise ValueError(
            f"ud_gguf_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_iq_gguf_format(value: object) -> str:
    """Validate ``value`` is an IQ GGUF format string. Returns canonical form."""
    _basic_validate(value, "iq_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in IQ_GGUF_FORMATS:
        supported = ", ".join(sorted(IQ_GGUF_FORMATS))
        raise ValueError(
            f"iq_gguf_format {value!r} not supported. Supported: {supported}"
        )
    return canonical


def validate_apple_arm_gguf_format(value: object) -> str:
    """Validate ``value`` is an Apple/ARM GGUF format string."""
    _basic_validate(value, "apple_arm_gguf_format")
    canonical = _resolve_canonical(value)  # type: ignore[arg-type]
    if canonical is None or canonical not in APPLE_ARM_GGUF_FORMATS:
        supported = ", ".join(sorted(APPLE_ARM_GGUF_FORMATS))
        raise ValueError(
            f"apple_arm_gguf_format {value!r} not supported. "
            f"Supported: {supported}"
        )
    return canonical


def is_ud_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the UD GGUF ladder entries."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in UD_GGUF_FORMATS


def is_iq_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the IQ GGUF flavours."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in IQ_GGUF_FORMATS


def is_apple_arm_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is one of the Apple/ARM GGUF flavours."""
    if isinstance(value, bool) or not isinstance(value, str):
        return False
    canonical = _resolve_canonical(value)
    return canonical is not None and canonical in APPLE_ARM_GGUF_FORMATS


def is_advanced_gguf_format(value: object) -> bool:
    """Return True iff ``value`` is any v0.53.0 advanced GGUF format."""
    return (
        is_ud_gguf_format(value)
        or is_iq_gguf_format(value)
        or is_apple_arm_gguf_format(value)
    )


def get_gguf_spec(name: str) -> GGUFQuantSpec:
    """Return the frozen :class:`GGUFQuantSpec` for ``name`` (case-insensitive)."""
    if isinstance(name, bool) or not isinstance(name, str):
        raise TypeError(f"name must be str, got {type(name).__name__}")
    canonical = _resolve_canonical(name)
    if canonical is None:
        supported_n = len(ALL_ADVANCED_GGUF_FORMATS)
        raise ValueError(
            f"GGUF format {name!r} not in v0.53.0 catalog "
            f"({supported_n} known)"
        )
    return _GGUF_METADATA[canonical]


def validate_calibration_data_path(path: object) -> str:
    """Validate ``--calibration-data <jsonl>`` argument shape.

    Boundary contract — what's enforced HERE vs at CLI dispatch (v0.53.1):

    THIS helper enforces:
    * non-empty ``str`` (bool / None / other types rejected with ``TypeError``)
    * no null bytes
    * length <= 4096 chars

    CLI dispatch in v0.53.1 MUST additionally apply (mirrors v0.43.0 /
    v0.46.0 / v0.47.0 TOCTOU policy):
    * ``os.path.realpath`` + ``os.path.commonpath`` cwd containment
    * ``os.lstat`` + ``stat.S_ISLNK`` rejection BEFORE any ``open()``
    * existence check via ``os.path.isfile``

    Do NOT skip the dispatch-time controls — this helper is shape-only.
    """
    if isinstance(path, bool):
        raise TypeError(f"calibration_data must not be bool, got {path!r}")
    if not isinstance(path, str):
        raise TypeError(
            f"calibration_data must be str, got {type(path).__name__}"
        )
    if not path:
        raise ValueError("calibration_data must be non-empty")
    if "\x00" in path:
        raise ValueError("calibration_data must not contain null bytes")
    if len(path) > 4096:
        raise ValueError("calibration_data path too long (max 4096 chars)")
    return path


# --- v0.53.1 #139 — Live llama.cpp imatrix + quantize wiring ---------------

# Max 30 min per subprocess so we don't hang CI forever on a bad build.
_SUBPROC_TIMEOUT_SECONDS: int = 30 * 60


def _safe_stderr(stderr: Optional[str], cap: int = 512) -> str:
    """Truncate + Rich-markup-escape subprocess stderr before embedding it
    in ``RuntimeError`` messages.

    Security review L4 — the llama-imatrix / llama-quantize binaries may
    echo crafted input back in their stderr; without escape, characters
    like ``[red]`` would inject Rich markup when the wrapping exception
    is later printed via ``console.print``.
    """
    if not stderr:
        return ""
    from rich.markup import escape

    truncated = stderr[:cap]
    return escape(truncated)

# Quantize flavours that require an imatrix file (UD ladder + low-bit IQ family).
_REQUIRES_IMATRIX: frozenset[str] = (
    UD_GGUF_FORMATS
    | frozenset({"IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ2_M",
                 "IQ3_XXS", "IQ3_XS"})
)


def _enforce_under_cwd_and_no_symlink(path: str, field: str) -> str:
    """Re-export shared helper from :mod:`soup_cli.utils.paths`."""
    from soup_cli.utils.paths import enforce_under_cwd_and_no_symlink

    return enforce_under_cwd_and_no_symlink(path, field)


def _resolve_quantize_binary(llama_cpp_dir: str) -> Path:
    """Locate the ``llama-quantize`` (or legacy ``quantize``) binary."""
    base = Path(llama_cpp_dir)
    candidates = [
        base / "llama-quantize",
        base / "llama-quantize.exe",
        base / "build" / "bin" / "llama-quantize",
        base / "build" / "bin" / "llama-quantize.exe",
        base / "quantize",
        base / "quantize.exe",
        base / "build" / "bin" / "quantize",
        base / "build" / "bin" / "quantize.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"llama-quantize binary not found under {llama_cpp_dir!r}. "
        "Build llama.cpp with cmake first."
    )


def _resolve_imatrix_binary(llama_cpp_dir: str) -> Path:
    """Locate the ``llama-imatrix`` (or legacy ``imatrix``) binary."""
    base = Path(llama_cpp_dir)
    candidates = [
        base / "llama-imatrix",
        base / "llama-imatrix.exe",
        base / "build" / "bin" / "llama-imatrix",
        base / "build" / "bin" / "llama-imatrix.exe",
        base / "imatrix",
        base / "imatrix.exe",
        base / "build" / "bin" / "imatrix",
        base / "build" / "bin" / "imatrix.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"llama-imatrix binary not found under {llama_cpp_dir!r}. "
        "Build llama.cpp tools with `cmake --build . --target llama-imatrix` "
        "(or `-DLLAMA_BUILD_TOOLS=ON` then `cmake --build .`)."
    )


def _prepare_calibration_text(calibration_data: str, staged_dir: Path) -> Path:
    """Read JSONL ``{"text": "..."}`` rows and write a plain-text file.

    llama.cpp ``imatrix`` accepts a raw text file (one paragraph per line is
    fine). We extract the ``text`` field from each JSONL row, dropping any
    row without a string ``text``.
    """
    src = Path(calibration_data)
    if not src.is_file():
        raise FileNotFoundError(
            f"calibration_data file not found: {os.path.basename(calibration_data)!r}"
        )
    # Security review M3 — defend against TOCTOU swap between the CLI-level
    # symlink check and this open(). Use O_NOFOLLOW on POSIX so a symlink
    # placed between the two calls is rejected at the kernel level.
    # On Windows there is no O_NOFOLLOW; the dispatch-time check from
    # ``enforce_under_cwd_and_no_symlink`` is the portable backstop.
    if hasattr(os, "O_NOFOLLOW"):
        try:
            fd = os.open(str(src), os.O_RDONLY | os.O_NOFOLLOW)
        except OSError as exc:
            raise ValueError(
                "calibration_data became a symlink during the export "
                "(TOCTOU defence): refusing to open."
            ) from exc
        os.close(fd)
    out = staged_dir / "calib.txt"
    line_count = 0
    total_bytes = 0
    max_per_line = 8192
    max_total_bytes = 50 * 1024 * 1024  # 50 MB cap on the rendered calib file

    def _sanitise(text: str) -> str:
        # Strip null bytes + collapse newlines to spaces; cap per-line length.
        sanitised = text.replace("\x00", "").replace("\n", " ")
        return sanitised[:max_per_line]

    with open(src, encoding="utf-8") as fh_in, open(
        out, "w", encoding="utf-8"
    ) as fh_out:
        for raw_line in fh_in:
            line = raw_line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Treat as a raw text line — sanitise the same way.
                emitted = _sanitise(line)
                if emitted:
                    fh_out.write(emitted + "\n")
                    line_count += 1
                    total_bytes += len(emitted) + 1
                continue
            if isinstance(row, dict):
                text = row.get("text") or row.get("prompt") or row.get("content")
            elif isinstance(row, str):
                text = row
            else:
                text = None
            if isinstance(text, str) and text:
                emitted = _sanitise(text)
                if emitted:
                    fh_out.write(emitted + "\n")
                    line_count += 1
                    total_bytes += len(emitted) + 1
            if line_count >= 4096 or total_bytes >= max_total_bytes:
                # Safety cap — imatrix doesn't need more than a few thousand
                break
    if line_count == 0:
        raise ValueError(
            "calibration_data produced 0 usable rows; "
            "expected JSONL with a 'text' field."
        )
    return out


def _run_convert_to_f16(
    llama_cpp_dir: str, model_dir: str, f16_out: str,
) -> None:
    """Invoke ``convert_hf_to_gguf.py`` to produce an f16 GGUF."""
    script = Path(llama_cpp_dir) / "convert_hf_to_gguf.py"
    if not script.is_file():
        raise FileNotFoundError(
            f"convert_hf_to_gguf.py not found in {llama_cpp_dir!r}"
        )
    # Security review M5 — defend against a crafted llama_cpp_dir whose
    # ``convert_hf_to_gguf.py`` is a symlink escaping the directory. Resolve
    # both paths to realpath and require the script to stay inside.
    script_real = os.path.realpath(str(script))
    dir_real = os.path.realpath(str(llama_cpp_dir))
    try:
        common = os.path.commonpath([script_real, dir_real])
    except ValueError:
        common = ""
    if common != dir_real:
        raise FileNotFoundError(
            "convert_hf_to_gguf.py is outside the llama.cpp dir "
            "(symlink escape rejected)"
        )

    argv = [
        sys.executable,
        script_real,
        str(model_dir),
        "--outfile", str(f16_out),
        "--outtype", "f16",
    ]
    result = subprocess.run(  # noqa: S603 — argv list, no shell
        argv, shell=False, timeout=_SUBPROC_TIMEOUT_SECONDS,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"convert_hf_to_gguf.py failed (rc={result.returncode}): "
            f"{_safe_stderr(result.stderr)}"
        )


def _run_imatrix(
    *,
    llama_cpp_dir: str,
    f16_path: str,
    calib_data: str,
    imatrix_out: str,
) -> None:
    """Run llama.cpp ``imatrix`` to compute an importance matrix."""
    binary = _resolve_imatrix_binary(llama_cpp_dir)
    argv = [
        str(binary),
        "-m", str(f16_path),
        "-f", str(calib_data),
        "-o", str(imatrix_out),
        "--chunks", "32",
    ]
    result = subprocess.run(  # noqa: S603 — argv list, no shell
        argv, shell=False, timeout=_SUBPROC_TIMEOUT_SECONDS,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-imatrix failed (rc={result.returncode}): "
            f"{_safe_stderr(result.stderr)}"
        )


def _flavour_to_quantize_arg(flavour: str) -> str:
    """Map a v0.53.0 flavour string to the llama.cpp ``quantize`` CLI arg.

    UD ladder strips the ``UD-`` prefix (llama.cpp doesn't know UD; the UD
    flavour is the underlying type + imatrix calibration). IQ + Apple/ARM
    pass through verbatim.
    """
    if flavour.startswith("UD-"):
        return flavour[len("UD-"):]
    return flavour


def _run_quantize_binary(
    *,
    llama_cpp_dir: str,
    f16_path: str,
    output_path: str,
    flavour: str,
    imatrix_path: Optional[str] = None,
) -> None:
    """Run llama.cpp ``quantize`` to write the final GGUF."""
    binary = _resolve_quantize_binary(llama_cpp_dir)
    argv: list[str] = [str(binary)]
    if imatrix_path is not None:
        argv += ["--imatrix", str(imatrix_path)]
    argv += [str(f16_path), str(output_path), _flavour_to_quantize_arg(flavour)]
    result = subprocess.run(  # noqa: S603 — argv list, no shell
        argv, shell=False, timeout=_SUBPROC_TIMEOUT_SECONDS,
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-quantize failed (rc={result.returncode}): "
            f"{_safe_stderr(result.stderr)}"
        )


def export_advanced_gguf(
    *,
    model_dir: str,
    output_path: str,
    flavour: str,
    calibration_data: Optional[str],
    llama_cpp_dir: str,
) -> None:
    """Export a HuggingFace model as a UD / IQ / Apple-ARM GGUF.

    Three-stage pipeline:
    1. ``convert_hf_to_gguf.py`` → ``f16.gguf``
    2. If ``flavour`` needs an importance matrix
       (UD ladder + low-bit IQ family): ``imatrix`` → ``imatrix.dat``
    3. ``quantize`` (with ``--imatrix`` when present) → ``output_path``

    All subprocess invocations use argv-list form (no shell). Per the
    v0.53.0 ``validate_calibration_data_path`` docstring contract, this
    dispatch-time helper applies cwd containment + symlink rejection.
    """
    # Flavour validation
    if not is_advanced_gguf_format(flavour):
        raise ValueError(
            f"Unknown gguf flavour {flavour!r}. "
            "See ALL_ADVANCED_GGUF_FORMATS."
        )

    _enforce_under_cwd_and_no_symlink(model_dir, "model_dir")
    _enforce_under_cwd_and_no_symlink(output_path, "output_path")
    _enforce_under_cwd_and_no_symlink(llama_cpp_dir, "llama_cpp_dir")

    if not os.path.isdir(model_dir):
        raise FileNotFoundError(
            f"model_dir not a directory: {os.path.basename(model_dir)!r}"
        )
    if not os.path.isdir(llama_cpp_dir):
        raise FileNotFoundError(
            f"llama_cpp_dir not a directory: "
            f"{os.path.basename(llama_cpp_dir)!r}"
        )

    needs_imatrix = flavour in _REQUIRES_IMATRIX
    if needs_imatrix and calibration_data is None:
        raise ValueError(
            f"flavour {flavour!r} requires --calibration-data <jsonl>. "
            "UD ladder + low-bit IQ flavours need an importance matrix."
        )
    if calibration_data is not None:
        _enforce_under_cwd_and_no_symlink(
            calibration_data, "calibration_data",
        )

    # Stage intermediate files inside a tempdir under cwd
    with tempfile.TemporaryDirectory(
        prefix=".soup_gguf_", dir=str(Path.cwd()),
    ) as staged:
        staged_path = Path(staged)
        f16_path = staged_path / "model.f16.gguf"

        # 1. Convert HF → f16 GGUF
        _run_convert_to_f16(llama_cpp_dir, model_dir, str(f16_path))

        # 2. Compute importance matrix (imatrix) — only for UD ladder + low-bit IQ
        imatrix_path: Optional[str] = None
        if needs_imatrix:
            # cast() rather than assert — survives `python -O`
            calib_data_str = cast(str, calibration_data)
            calib_txt = _prepare_calibration_text(calib_data_str, staged_path)
            imatrix_path = str(staged_path / "imatrix.dat")
            _run_imatrix(
                llama_cpp_dir=llama_cpp_dir,
                f16_path=str(f16_path),
                calib_data=str(calib_txt),
                imatrix_out=imatrix_path,
            )

        # 3. Final quantize
        _run_quantize_binary(
            llama_cpp_dir=llama_cpp_dir,
            f16_path=str(f16_path),
            output_path=output_path,
            flavour=flavour,
            imatrix_path=imatrix_path,
        )

    # Ensure the writer produced the file
    if not os.path.isfile(output_path):
        raise RuntimeError(
            f"llama-quantize did not produce {os.path.basename(output_path)!r}"
        )
