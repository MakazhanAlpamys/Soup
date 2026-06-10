"""v0.71.20 #131 — Live TTS (text-to-speech) fine-tuning trainer.

``TTSTrainerWrapper`` lifts the v0.52.0 ``build_tts_trainer`` schema stub. The
five upstream TTS families (orpheus / sesame_csm / llasa / spark / oute) are
all decoder language models. A TTS fine-tune trains the LM with next-token
cross-entropy over interleaved ``[text][audio-codec-token]`` sequences — the
exact objective the SFT trainer already implements. So this wrapper reuses
:class:`~soup_cli.trainer.sft.SFTTrainerWrapper` for model/tokenizer/LoRA/CE
and layers TTS-specific behaviour on top:

* **Pre-encoded chat mode** (live, validated): the operator runs the family's
  audio codec OFFLINE, so the assistant turn already carries the codec-token
  string. Training is then plain SFT cross-entropy and runs on any GPU.
  Soup's per-family contribution here is (a) emotion-control templating for
  emotion-conditioned families and (b) registration of operator-supplied
  codec special tokens (``data.new_special_tokens``) with an embedding resize.
* **Live-codec mode** (``data.format == 'audio'``): encoding raw audio into
  codec tokens at train time needs the family's heavyweight codec dependency
  (SNAC / BiCodec / XCodec2 / …). This path is hardware/dependency-gated and
  raises a friendly per-family ``RuntimeError`` naming the required package.

Security / robustness:
- Family validated via :func:`soup_cli.utils.tts.validate_tts_family` before
  any model load.
- Per-family codec gate fails loud with an actionable ``pip install`` hint.
- Special-token registration only adds tokens not already in the vocab and
  resizes embeddings through the (possibly PEFT-wrapped) model.
"""

from __future__ import annotations

from typing import Optional

from soup_cli.trainer.sft import SFTTrainerWrapper, console
from soup_cli.utils.tts import (
    format_tts_messages,
    tts_codec_package,
    validate_tts_family,
)

# data.format values that require encoding raw audio at train time.
_LIVE_CODEC_FORMATS: frozenset[str] = frozenset({"audio"})


class TTSTrainerWrapper(SFTTrainerWrapper):
    """SFT-based TTS trainer (v0.71.20 #131).

    Inherits the full SFT setup/train machinery (Quant Menu model load, LoRA,
    chat formatting, cross-entropy, callbacks). Overrides :meth:`setup` to
    validate the family, gate the live-codec path, and emotion-template the
    chat data, and :meth:`_setup_transformers` to register codec special
    tokens after the model loads.
    """

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._tts_family: Optional[str] = None

    def setup(self, dataset: dict) -> None:
        cfg = self.config
        tcfg = cfg.training
        family = validate_tts_family(tcfg.tts_family)
        self._tts_family = family

        data_format = getattr(cfg.data, "format", None)
        if data_format in _LIVE_CODEC_FORMATS:
            # Live-codec mode: encoding raw audio needs the family's codec.
            self._require_tts_codec(family)
            # If the codec IS importable (never validated on the maintainer's
            # box — no codecs/large models installed), the encode-then-CE path
            # would run here. It is intentionally surfaced as an explicit
            # not-yet-validated error so we never silently ship an unrun path.
            pkg = tts_codec_package(family)
            raise RuntimeError(
                f"TTS live-codec mode (data.format='audio') for family "
                f"'{family}' requires encoding audio with the {pkg!r} codec at "
                "train time. This path is hardware/dependency-gated and not "
                "yet validated. Use the pre-encoded chat workflow (encode "
                "audio to codec tokens offline, then train with "
                "data.format=chat) for a runnable TTS fine-tune."
            )

        # Pre-encoded chat mode: plain SFT cross-entropy over the codec tokens.
        console.print(
            f"[green]TTS fine-tune:[/] family={family} (pre-encoded chat mode, "
            "next-token CE)"
        )
        dataset = self._apply_tts_templating(dataset, family, tcfg.tts_emotion)
        super().setup(dataset)

    def _require_tts_codec(self, family: str) -> None:
        """Raise a friendly per-family error when the live codec is absent."""
        import importlib.util

        pkg = tts_codec_package(family)
        if importlib.util.find_spec(pkg) is None:
            raise RuntimeError(
                f"TTS family '{family}' live-codec training requires the "
                f"'{pkg}' package (audio codec). Install it with "
                f"`pip install {pkg}`, or pre-encode your audio to codec "
                "tokens offline and train with data.format=chat."
            )

    def _apply_tts_templating(
        self, dataset: dict, family: str, emotion: Optional[str]
    ) -> dict:
        """Return a new dataset dict with per-family emotion templating."""

        def _map_split(rows: object) -> list:
            if not isinstance(rows, (list, tuple)):
                return []
            out = []
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("messages"), (list, tuple)):
                    new_row = dict(row)
                    new_row["messages"] = format_tts_messages(
                        row["messages"], family, emotion=emotion
                    )
                    out.append(new_row)
                else:
                    out.append(row)
            return out

        new_dataset = dict(dataset)
        if "train" in new_dataset:
            new_dataset["train"] = _map_split(new_dataset["train"])
        if new_dataset.get("val"):
            new_dataset["val"] = _map_split(new_dataset["val"])
        return new_dataset

    def _setup_transformers(self, cfg, tcfg) -> None:  # type: ignore[override]
        # NOTE: this override fires because TTS pins modality='audio_out'
        # (validate_tts_compat), which is neither 'vision' nor 'audio', so
        # SFTTrainerWrapper.setup dispatches to the text path here. If a future
        # schema change ever let TTS run with modality='audio', the dispatch
        # would go to _setup_audio_transformers and this hook would be skipped.
        super()._setup_transformers(cfg, tcfg)
        self._register_tts_special_tokens(cfg)

    def _register_tts_special_tokens(self, cfg) -> None:
        """Register operator-supplied codec special tokens + resize embeds.

        Reuses the v0.42.0 ``data.new_special_tokens`` field. Only tokens not
        already present are added (deduplicated, order-preserving); the
        embedding matrix is resized through the (possibly PEFT-wrapped) model
        so the new codec-token ids have rows.
        """
        new_tokens = getattr(cfg.data, "new_special_tokens", None)
        if not new_tokens:
            return
        existing = set(self.tokenizer.get_vocab().keys())
        # dict.fromkeys dedups within new_tokens while preserving order.
        to_add = [t for t in dict.fromkeys(new_tokens) if t not in existing]
        if not to_add:
            return
        added = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": to_add}
        )
        if added:
            self.model.resize_token_embeddings(len(self.tokenizer))
            console.print(
                f"[green]TTS special tokens:[/] registered {added} codec "
                "token(s) + resized embeddings"
            )
