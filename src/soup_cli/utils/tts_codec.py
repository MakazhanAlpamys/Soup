"""v0.71.22 #265-partial — live-codec TTS audio encoding (Orpheus via SNAC).

Lifts the v0.71.20 ``data.format='audio'`` hardware gate into a real
encode-at-train-time path. Rows shaped ``{"audio": <path>, "messages": [...]}``
(the v0.17.0 audio format — paths are resolved + containment-checked by the
data loader) are encoded into discrete codec-token strings that become the
assistant turn, after which training proceeds through the validated
pre-encoded SFT cross-entropy path.

The generic pipeline (``encode_tts_dataset`` → ``encode_tts_row`` →
per-family encoder) is family-agnostic; v0.71.22 ships ONE validated encoder:

* **Orpheus / SNAC** (``pip install snac``) — 24 kHz SNAC produces 3
  codebooks at a 1/2/4 frame ratio; Orpheus interleaves them 7 tokens per
  frame with per-slot codebook offsets and renders each code as
  ``<custom_token_N>`` where ``N = code + 10 + slot*4096`` (the official
  Orpheus id layout: ``id = 128256 + N = 128266 + code + slot*4096`` on the
  Orpheus tokenizer, whose ``<custom_token_i>`` maps to ``128256 + i``).

The other four families (sesame_csm / llasa / spark / oute) keep their
per-family codec dep gate; their encoders are tracked in #265 and raise a
friendly ``RuntimeError`` pointing at the offline pre-encode workflow.

Security / robustness:
- Heavy imports (numpy / soundfile / torch / snac) are lazy — module import
  stays light for the CLI hot path.
- Audio paths: null-byte rejection, symlink rejection (``os.lstat``,
  defence-in-depth — the loader already containment-checks under
  ``data.audio_dir``), duration cap (``_MAX_AUDIO_SECONDS``).
- Pure interleave kernel validates the 1/2/4 codebook ratio, per-code bounds
  ``[0, 4096)``, and rejects bool codes (project bool-as-int policy).
- Rows are deep-copied — the caller's dataset is never mutated (v0.33.0 #47
  immutability policy).
"""

from __future__ import annotations

import os
import stat
from typing import Any, Callable, Optional

from soup_cli.utils.tts import tts_codec_package, validate_tts_family

# SNAC checkpoint for the Orpheus family (24 kHz, 3 codebooks).
SNAC_MODEL_ID = "hubertsiuzdak/snac_24khz"
SNAC_SAMPLE_RATE = 24_000

# Orpheus token layout: <custom_token_N> with N = code + OFFSET + slot*4096.
# Slots 0..9 are Orpheus control tokens; audio codes start at 10.
ORPHEUS_CODE_OFFSET = 10
ORPHEUS_CODEBOOK_SIZE = 4096
_ORPHEUS_FRAME_SLOTS = 7

# Families whose live-codec encoder is implemented AND validated.
LIVE_CODEC_FAMILIES: frozenset[str] = frozenset({"orpheus"})

_MAX_AUDIO_SECONDS = 600.0  # 10 min cap — defends against runaway encodes
# Byte cap (~600 s of 24 kHz stereo float32 + headroom) — rejected before any
# read so a runaway file never materialises into RAM.
_MAX_AUDIO_BYTES = 512 * 1024 * 1024

# Process-level SNAC model cache (one load per device). NOTE: single-threaded
# assumption — the live-codec trainer encode path runs on one thread, so this
# unbounded, unguarded cache is intentional (no eviction / no lock needed).
_SNAC_CACHE: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Pure interleave kernel (no torch — unit-testable on plain lists)
# ---------------------------------------------------------------------------


def _check_codes(codes, name: str, expected_len: int) -> list[int]:
    out = []
    for code in codes:
        if isinstance(code, bool) or not isinstance(code, int):
            raise TypeError(f"{name} codes must be non-bool ints")
        if not (0 <= code < ORPHEUS_CODEBOOK_SIZE):
            raise ValueError(
                f"{name} code {code} out of range [0, {ORPHEUS_CODEBOOK_SIZE})"
            )
        out.append(code)
    if len(out) != expected_len:
        raise ValueError(
            f"{name} codebook length {len(out)} != expected {expected_len} "
            "(SNAC codebooks must have the 1/2/4 frame ratio)"
        )
    return out


def interleave_orpheus_codes(coarse, medium, fine) -> list[int]:
    """Interleave the 3 SNAC codebooks into Orpheus token indices.

    ``coarse``/``medium``/``fine`` are the per-frame code sequences at the
    SNAC 1/2/4 ratio (lengths T / 2T / 4T). Returns 7 token indices per
    frame in the official Orpheus slot order, each offset by
    ``ORPHEUS_CODE_OFFSET + slot * ORPHEUS_CODEBOOK_SIZE`` so the rendered
    ``<custom_token_N>`` strings map onto the Orpheus tokenizer's audio ids.
    """
    coarse = list(coarse)
    if not coarse:
        raise ValueError("coarse codebook must be non-empty")
    frames = len(coarse)
    coarse = _check_codes(coarse, "coarse", frames)
    medium = _check_codes(list(medium), "medium", 2 * frames)
    fine = _check_codes(list(fine), "fine", 4 * frames)

    size = ORPHEUS_CODEBOOK_SIZE
    base = ORPHEUS_CODE_OFFSET
    out: list[int] = []
    for i in range(frames):
        out.append(coarse[i] + base)
        out.append(medium[2 * i] + base + size)
        out.append(fine[4 * i] + base + 2 * size)
        out.append(fine[4 * i + 1] + base + 3 * size)
        out.append(medium[2 * i + 1] + base + 4 * size)
        out.append(fine[4 * i + 2] + base + 5 * size)
        out.append(fine[4 * i + 3] + base + 6 * size)
    return out


def orpheus_tokens_to_string(indices) -> str:
    """Render Orpheus token indices as a ``<custom_token_N>`` string.

    Indices must be non-bool ints (mirrors :func:`_check_codes` — project
    bool-as-int policy; no float/bool coercion).
    """
    indices = list(indices)
    if not indices:
        raise ValueError("token index list must be non-empty")
    for n in indices:
        if isinstance(n, bool) or not isinstance(n, int):
            raise TypeError("token indices must be non-bool ints")
    return "".join(f"<custom_token_{n}>" for n in indices)


# ---------------------------------------------------------------------------
# Audio loading (lazy soundfile / numpy)
# ---------------------------------------------------------------------------


def load_audio_mono(path: str, *, target_sr: int = SNAC_SAMPLE_RATE):
    """Load an audio file as mono float32 at ``target_sr``.

    Lazy-imports soundfile (friendly ImportError naming the install).
    Off-rate audio is resampled with basic linear interpolation — adequate
    for codec encoding of speech; operators wanting studio-grade resampling
    should resample offline. Returns a 1-D ``np.float32`` array.
    """
    if not isinstance(path, str) or not path:
        raise ValueError("audio path must be a non-empty string")
    if "\x00" in path:
        raise ValueError("audio path must not contain null bytes")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"audio file not found: {os.path.basename(path)!r}"
        )
    # Symlink rejection — defence-in-depth; the loader already containment-
    # checks rows under data.audio_dir (v0.17.0 policy).
    if stat.S_ISLNK(os.lstat(path).st_mode):
        raise ValueError("audio path must not be a symlink")
    # Byte-size cap before any read — defends against a runaway file.
    if os.path.getsize(path) > _MAX_AUDIO_BYTES:
        raise ValueError(
            f"audio file exceeds the {_MAX_AUDIO_BYTES} byte live-codec cap; "
            "split the clip or pre-encode offline"
        )
    try:
        import soundfile
    except ImportError as exc:  # pragma: no cover - dep present in CI
        raise ImportError(
            "Live-codec TTS needs the 'soundfile' package to read audio. "
            "Install it with `pip install soundfile` (or `pip install "
            "'soup-cli[audio]'`)."
        ) from exc
    import numpy as np

    # Probe metadata FIRST — reject an over-long clip before materialising the
    # whole file into RAM (the post-read check below stays as defence-in-depth).
    info = soundfile.info(path)
    if info.samplerate and info.frames / float(info.samplerate) > _MAX_AUDIO_SECONDS:
        raise ValueError(
            f"audio is {info.frames / float(info.samplerate):.1f} seconds — "
            f"exceeds the {_MAX_AUDIO_SECONDS:.0f} seconds live-codec cap; "
            "split the clip or pre-encode offline"
        )
    # Open via O_NOFOLLOW and hand the fd to soundfile — closes the TOCTOU
    # window between the lstat symlink check above and the read (project
    # v0.65/v0.67 O_NOFOLLOW reader policy). soundfile.read accepts an open
    # file object.
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        with os.fdopen(fd, "rb") as handle:
            data, sr = soundfile.read(handle, dtype="float32", always_2d=True)
    except OSError as exc:
        # O_NOFOLLOW raises ELOOP if the path became a symlink after the lstat.
        raise ValueError(f"audio path could not be read: {exc}") from exc
    duration = data.shape[0] / float(sr)
    if duration > _MAX_AUDIO_SECONDS:
        raise ValueError(
            f"audio is {duration:.1f} seconds — exceeds the "
            f"{_MAX_AUDIO_SECONDS:.0f} seconds live-codec cap; split the "
            "clip or pre-encode offline"
        )
    mono = data.mean(axis=1)
    if int(sr) != int(target_sr):
        n_out = max(1, int(round(len(mono) * target_sr / float(sr))))
        x_old = np.linspace(0.0, 1.0, num=len(mono), endpoint=False)
        x_new = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
        mono = np.interp(x_new, x_old, mono)
    return mono.astype(np.float32)


# ---------------------------------------------------------------------------
# Orpheus / SNAC encoder
# ---------------------------------------------------------------------------


def _get_snac_model(device: Optional[str] = None):
    """Load (once per device) the SNAC codec model. Lazy snac import."""
    dev = device or "cpu"
    cached = _SNAC_CACHE.get(dev)
    if cached is not None:
        return cached
    try:
        from snac import SNAC
    except ImportError as exc:
        raise ImportError(
            "TTS family 'orpheus' live-codec training requires the 'snac' "
            "package (audio codec). Install it with `pip install snac`, or "
            "pre-encode your audio to codec tokens offline and train with "
            "data.format=chat."
        ) from exc
    model = SNAC.from_pretrained(SNAC_MODEL_ID).eval().to(dev)
    _SNAC_CACHE[dev] = model
    return model


def encode_audio_orpheus(
    path: str,
    *,
    snac_model: Optional[Any] = None,
    device: Optional[str] = None,
) -> str:
    """Encode one audio file into the Orpheus ``<custom_token_N>`` string.

    ``snac_model`` is injectable (tests / pre-loaded models); when ``None``
    the 24 kHz SNAC checkpoint is loaded once per process per device.
    """
    audio = load_audio_mono(path, target_sr=SNAC_SAMPLE_RATE)
    import torch

    model = snac_model if snac_model is not None else _get_snac_model(device)
    wav = torch.from_numpy(audio).reshape(1, 1, -1)
    if device:
        wav = wav.to(device)
    with torch.no_grad():
        codes = model.encode(wav)
    coarse = [int(c) for c in codes[0][0].tolist()]
    medium = [int(c) for c in codes[1][0].tolist()]
    fine = [int(c) for c in codes[2][0].tolist()]
    return orpheus_tokens_to_string(
        interleave_orpheus_codes(coarse, medium, fine)
    )


def tts_encoder_for_family(
    family: str, *, device: Optional[str] = None
) -> Callable[[str], str]:
    """Return the live audio→codec-string encoder for ``family``.

    Only the families in :data:`LIVE_CODEC_FAMILIES` have a validated
    encoder (v0.71.22 ships Orpheus/SNAC). The remaining families raise a
    friendly ``RuntimeError`` naming the offline workflow — their encoders
    are tracked in #265 (the per-family codec dep gate fires earlier, at
    trainer setup).
    """
    canonical = validate_tts_family(family)
    if canonical == "orpheus":

        def _encode(path: str) -> str:
            return encode_audio_orpheus(path, device=device)

        return _encode
    pkg = tts_codec_package(canonical)
    raise RuntimeError(
        f"Live-codec encoding for TTS family '{canonical}' is not yet "
        f"implemented (v0.71.22 ships the Orpheus/SNAC encoder; the "
        f"'{canonical}' encoder via {pkg!r} is tracked in #265). Pre-encode "
        "your audio to codec tokens offline and train with data.format=chat."
    )


# ---------------------------------------------------------------------------
# Dataset mapping
# ---------------------------------------------------------------------------


def encode_tts_row(row: dict, encoder: Callable[[str], str]) -> dict:
    """Encode one ``{"audio", "messages"}`` row into a chat row.

    The codec-token string becomes the FINAL assistant turn: an existing
    trailing assistant message has its content replaced (the audio is the
    training target), otherwise an assistant turn is appended. Returns a NEW
    deep-copied row without the ``audio`` key — the caller's row is never
    mutated.
    """
    import copy

    if not isinstance(row, dict):
        raise TypeError(f"row must be a dict, got {type(row).__name__}")
    audio = row.get("audio")
    if not isinstance(audio, str) or not audio:
        raise ValueError("live-codec TTS row must have a non-empty 'audio' path")
    messages = row.get("messages")
    if not isinstance(messages, (list, tuple)):
        raise ValueError("live-codec TTS row must have a 'messages' list")

    codec_string = encoder(audio)
    new_messages = [copy.deepcopy(m) for m in messages]
    if new_messages and isinstance(new_messages[-1], dict) and (
        new_messages[-1].get("role") == "assistant"
    ):
        new_messages[-1]["content"] = codec_string
    else:
        new_messages.append({"role": "assistant", "content": codec_string})
    out = {
        key: copy.deepcopy(value)
        for key, value in row.items()
        if key not in ("audio", "messages")
    }
    out["messages"] = new_messages
    return out


def encode_tts_dataset(
    dataset: dict,
    family: str,
    *,
    encoder: Optional[Callable[[str], str]] = None,
    device: Optional[str] = None,
    console: Optional[Any] = None,
) -> dict:
    """Encode every ``train`` / ``val`` row's audio into codec-token chat rows.

    Returns a NEW dataset dict (input never mutated). ``encoder`` is
    injectable for tests; by default it is resolved per-family via
    :func:`tts_encoder_for_family` (which raises for not-yet-implemented
    families — #265).
    """
    if not isinstance(dataset, dict):
        raise TypeError(
            f"dataset must be a dict, got {type(dataset).__name__}"
        )
    canonical = validate_tts_family(family)
    encode = encoder if encoder is not None else tts_encoder_for_family(
        canonical, device=device
    )

    new_dataset = dict(dataset)
    total = 0
    for split in ("train", "val"):
        rows = new_dataset.get(split)
        if not isinstance(rows, (list, tuple)):
            continue
        new_dataset[split] = [encode_tts_row(row, encode) for row in rows]
        total += len(rows)
    if console is not None:
        console.print(
            f"[green]TTS live-codec:[/] encoded {total} row(s) "
            f"(family={canonical})"
        )
    return new_dataset
