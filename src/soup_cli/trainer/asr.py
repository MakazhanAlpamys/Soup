"""v0.71.32 — ASR (Whisper) fine-tuning trainer.

``AsrTrainerWrapper`` trains ``WhisperForConditionalGeneration`` with a HF
``Seq2SeqTrainer`` — the sequence-to-sequence analogue of the classifier
wrapper (non-TRL: load model + processor, map the dataset, build a HF trainer).
It is the first Soup task whose input modality is raw audio.

Data (``data.format='asr'``): each row is ``{"audio": <path>, "text":
<transcript>}``. Audio is decoded to 16 kHz mono via the shared
``utils.tts_codec.load_audio_mono`` (soundfile pre-probe + O_NOFOLLOW read +
symlink / size guards), turned into log-mel ``input_features`` by the Whisper
feature extractor; the transcript is tokenized into decoder ``labels``.

Security / robustness:
- Arch guard (:func:`_require_whisper_base`) rejects a non-Whisper base BEFORE
  any weight download, naming the actual ``model_type``.
- ``data.audio_dir`` containment is enforced by the data loader; the decode
  path here adds the ``load_audio_mono`` symlink / size / TOCTOU guards.
- ``trust_remote_code`` threaded through the v0.36.0 resolver.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any

from rich.console import Console

from soup_cli.config.schema import SoupConfig, TrainingConfig

console = Console()

# Whisper always expects 16 kHz mono audio.
_ASR_SAMPLE_RATE: int = 16000

# Sidecar file recording the decode-time language/task so inference can restore
# them (WhisperProcessor.save_pretrained does NOT persist set_prefix_tokens —
# the v0.71.32 code-review CRITICAL).
_ASR_SIDECAR: str = "asr_generation.json"


def write_asr_sidecar(output_dir: str, language: str | None, task: str) -> None:
    """Persist the ASR decode prefix (language/task) next to the model."""
    import json

    payload = {"language": language, "task": task}
    with open(os.path.join(output_dir, _ASR_SIDECAR), "w", encoding="utf-8") as fh:
        json.dump(payload, fh)


def read_asr_sidecar(model_dir: str) -> dict:
    """Read the ASR decode prefix sidecar; ``{}`` when absent/unreadable.

    Values are shape-validated (a shared / downloaded model directory could
    carry a hostile sidecar): ``language`` must be a str <= 32 chars with no
    NUL; ``task`` must be one of transcribe/translate. Anything else is
    dropped so it never reaches ``whisper.generate``.
    """
    import json

    path = os.path.join(model_dir, _ASR_SIDECAR)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    out: dict = {}
    language = data.get("language")
    if isinstance(language, str) and language and "\x00" not in language and len(language) <= 32:
        out["language"] = language
    task = data.get("task")
    if task in ("transcribe", "translate"):
        out["task"] = task
    return out


def _validate_asr_row(row: dict) -> tuple[str, str]:
    """Extract ``(audio_path, transcript)`` from an ASR row.

    Raises:
        ValueError: missing ``audio`` / ``text`` or a non-string transcript.
        TypeError: a non-string ``audio`` value.
    """
    audio = row.get("audio")
    if audio is None or (isinstance(audio, str) and not audio.strip()):
        raise ValueError("ASR row must have a non-empty 'audio' path")
    if not isinstance(audio, str):
        raise TypeError(
            f"ASR row 'audio' must be a string path, got {type(audio).__name__}"
        )
    if "text" not in row:
        raise ValueError("ASR row must have a 'text' transcript")
    text = row["text"]
    if not isinstance(text, str):
        raise ValueError(
            f"ASR row 'text' must be a string, got {type(text).__name__}"
        )
    return audio, text


def _prefix_customized(asr_language: str | None, asr_task: str) -> bool:
    """True when the Whisper decoder prefix departs from the default.

    A bare ``asr_task='translate'`` (``asr_language=None``) must still customize
    the prefix — otherwise the translate objective silently trains/decodes as
    plain transcribe.
    """
    return bool(asr_language) or asr_task != "transcribe"


def _load_autoconfig(base: str, trust_remote_code: bool) -> Any:
    """Load ``AutoConfig`` for ``base`` (isolated for test monkeypatching)."""
    from transformers import AutoConfig

    return AutoConfig.from_pretrained(base, trust_remote_code=trust_remote_code)


def _require_whisper_base(base: str, trust_remote_code: bool) -> Any:
    """Return the model config iff ``base`` is a Whisper model, else raise.

    ``base`` is a free-form string so this cannot be a schema Literal — the
    guard runs at setup time, before the (potentially large) weights download,
    and names the actual ``model_type`` in the error.
    """
    cfg = _load_autoconfig(base, trust_remote_code)
    model_type = getattr(cfg, "model_type", None)
    if model_type != "whisper":
        raise ValueError(
            f"task='asr' requires a Whisper base model, but {base!r} has "
            f"model_type={model_type!r}. Use e.g. openai/whisper-tiny / "
            "whisper-base / whisper-large-v3."
        )
    return cfg


class AsrTrainerWrapper:
    """High-level Whisper ASR fine-tuning wrapper (v0.71.32)."""

    def __init__(
        self,
        config: SoupConfig,
        device: str = "cuda",
        report_to: str = "none",
        deepspeed_config: str | None = None,
        fsdp_config: dict | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        self.config = config
        self.device = device
        self.report_to = report_to
        self.deepspeed_config = deepspeed_config
        self.fsdp_config = fsdp_config

        from soup_cli.utils.trust_remote import (
            model_requires_trust_remote_code,
            resolve_trust_remote_code,
        )

        requires = model_requires_trust_remote_code(config.base) or False
        self._trust_remote_code = resolve_trust_remote_code(
            config.base,
            requested=trust_remote_code,
            console=console,
            requires_remote_code=requires,
        )

        self.model: Any = None
        self.processor: Any = None
        self.trainer: Any = None
        self._output_dir: str | None = None
        self._lora_active: bool = False
        self._prefix_customized: bool = False

    def setup(self, dataset: dict) -> None:
        """Load Whisper + processor, encode the dataset, build Seq2SeqTrainer."""
        from datasets import Dataset
        from transformers import (
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
            WhisperForConditionalGeneration,
            WhisperProcessor,
        )

        cfg = self.config
        tcfg = cfg.training

        # Arch guard BEFORE any weight download.
        _require_whisper_base(cfg.base, self._trust_remote_code)

        console.print(f"[dim]Loading Whisper processor: {cfg.base}[/]")
        self.processor = WhisperProcessor.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        # For fine-tuning, the decoder prefix (language / task) is baked into
        # the tokenized labels; forced_decoder_ids must be cleared so training
        # does not double-force them (HF Whisper fine-tuning guide). Fire when
        # EITHER knob departs from the default — a bare asr_task='translate'
        # (asr_language=None) must still set the translate prefix, not silently
        # train as transcribe.
        prefix_customized = _prefix_customized(tcfg.asr_language, tcfg.asr_task)
        self._prefix_customized = prefix_customized
        if prefix_customized:
            self.processor.tokenizer.set_prefix_tokens(
                language=tcfg.asr_language, task=tcfg.asr_task
            )

        console.print(f"[dim]Loading Whisper model: {cfg.base}[/]")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            cfg.base, trust_remote_code=self._trust_remote_code
        )
        self.model.config.forced_decoder_ids = None
        self.model.config.suppress_tokens = []
        # Store the decode-time forced ids for inference reuse.
        self._forced_decoder_ids = (
            self.processor.get_decoder_prompt_ids(
                language=tcfg.asr_language, task=tcfg.asr_task
            )
            if prefix_customized
            else None
        )

        # Optional LoRA on the attention q/v projections — opt-in via
        # ``training.asr_lora`` (default full-FT; tiny Whisper fits the dev box).
        if self._should_use_lora(tcfg):
            from peft import LoraConfig, get_peft_model

            target_modules = tcfg.lora.target_modules
            if target_modules == "auto":
                target_modules = ["q_proj", "v_proj"]
            lora_config = LoraConfig(
                r=tcfg.lora.r,
                lora_alpha=tcfg.lora.alpha,
                lora_dropout=tcfg.lora.dropout,
                target_modules=target_modules,
                bias="none",
                use_dora=tcfg.lora.use_dora,
                use_rslora=tcfg.lora.use_rslora,
            )
            self.model = get_peft_model(self.model, lora_config)
            self._lora_active = True
            console.print(
                f"[green]ASR LoRA enabled[/] (r={tcfg.lora.r}, "
                f"targets={target_modules})"
            )
        else:
            self._lora_active = False

        feature_extractor = self.processor.feature_extractor
        tokenizer = self.processor.tokenizer

        from soup_cli.utils.tts_codec import load_audio_mono

        def encode(row: dict) -> dict:
            audio_path, text = _validate_asr_row(row)
            wave = load_audio_mono(audio_path, target_sr=_ASR_SAMPLE_RATE)
            features = feature_extractor(
                wave, sampling_rate=_ASR_SAMPLE_RATE
            ).input_features[0]
            labels = tokenizer(text).input_ids
            return {"input_features": features, "labels": labels}

        raw_train = Dataset.from_list(dataset["train"])
        train_ds = raw_train.map(encode, remove_columns=raw_train.column_names)
        eval_ds = None
        if dataset.get("val"):
            raw_val = Dataset.from_list(dataset["val"])
            eval_ds = raw_val.map(encode, remove_columns=raw_val.column_names)

        output_dir = Path(cfg.output)
        if cfg.experiment_name:
            output_dir = output_dir / cfg.experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_size = tcfg.batch_size if tcfg.batch_size != "auto" else 8
        total_steps = (
            math.ceil(len(train_ds) / batch_size / tcfg.gradient_accumulation_steps)
            * tcfg.epochs
        )
        warmup_steps = int(total_steps * tcfg.warmup_ratio)

        args = Seq2SeqTrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=tcfg.epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=tcfg.gradient_accumulation_steps,
            learning_rate=tcfg.lr,
            warmup_steps=warmup_steps,
            weight_decay=tcfg.weight_decay,
            max_grad_norm=tcfg.max_grad_norm,
            optim=tcfg.optimizer,
            lr_scheduler_type=tcfg.scheduler,
            logging_steps=tcfg.logging_steps,
            save_steps=tcfg.save_steps,
            save_total_limit=3,
            bf16=self.device == "cuda",
            report_to=self.report_to,
            deepspeed=self.deepspeed_config,
            predict_with_generate=True,
            remove_unused_columns=False,
            **(self.fsdp_config or {}),
        )

        collator = _SpeechSeq2SeqCollator(
            self.processor,
            decoder_start_token_id=self._unwrapped_model().config.decoder_start_token_id,
        )
        self.trainer = Seq2SeqTrainer(
            model=self.model,
            args=args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            data_collator=collator,
            tokenizer=self.processor.feature_extractor,
        )
        self._output_dir = str(output_dir)

    def _unwrapped_model(self) -> Any:
        """Return the underlying Whisper model (unwrap a PEFT wrapper)."""
        model = self.model
        get_base = getattr(model, "get_base_model", None)
        return get_base() if callable(get_base) else model

    def _should_use_lora(self, tcfg: TrainingConfig) -> bool:
        """LoRA is on iff ``asr_lora`` is opted in AND rank > 0.

        Mirrors ``classifier_lora`` — a bare ``task: asr`` config must default
        to full fine-tune, not silently apply the schema-default rank-64 LoRA.
        """
        lora = getattr(tcfg, "lora", None)
        return (
            bool(getattr(tcfg, "asr_lora", False))
            and lora is not None
            and getattr(lora, "r", 0) > 0
        )

    def train(
        self,
        display: object | None = None,
        tracker: object | None = None,
        run_id: str = "",
        resume_from_checkpoint: str | None = None,
    ) -> dict:
        if self.trainer is None:
            raise RuntimeError(
                "AsrTrainerWrapper.train() called before setup(). "
                "Call setup(dataset) first."
            )
        start = time.time()
        if display is not None:
            from soup_cli.monitoring.callback import SoupTrainerCallback

            self.trainer.add_callback(
                SoupTrainerCallback(
                    display, tracker=tracker, run_id=run_id,
                    loss_watchdog=self.config.training.loss_watchdog,
                    loss_watchdog_threshold=self.config.training.loss_watchdog_threshold,
                    loss_watchdog_patience=self.config.training.loss_watchdog_patience,
                    eval_gate_config=self.config.training.eval_gate,
                )
            )
        self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        duration = time.time() - start

        self.trainer.save_model(self._output_dir)
        self.processor.save_pretrained(self._output_dir)
        # Persist language/task so `soup infer --task asr` restores them
        # (set_prefix_tokens is NOT serialized by the processor).
        if self._prefix_customized:
            write_asr_sidecar(
                self._output_dir,
                self.config.training.asr_language,
                self.config.training.asr_task,
            )

        logs = self.trainer.state.log_history
        train_losses = [entry["loss"] for entry in logs if "loss" in entry]

        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        duration_str = f"{hours}h {minutes}m" if hours > 0 else f"{minutes}m"
        return {
            "initial_loss": train_losses[0] if train_losses else 0,
            "final_loss": train_losses[-1] if train_losses else 0,
            "duration": duration_str,
            "duration_secs": duration,
            "output_dir": self._output_dir,
            "total_steps": self.trainer.state.global_step,
        }


def _strip_decoder_start(labels: Any, decoder_start_token_id: int | None) -> Any:
    """Drop a leading ``decoder_start_token_id`` column if every row has it.

    Whisper's ``tokenizer(text)`` prepends ``<|startoftranscript|>`` (=
    ``decoder_start_token_id``, e.g. 50258), and the model re-adds it via
    ``shift_tokens_right`` — so it MUST be stripped from the labels or every
    example teaches "after SOT predict SOT". This is keyed on
    ``decoder_start_token_id``, NOT ``bos_token_id`` (for Whisper ``bos`` ==
    ``eos`` == 50257, which never equals the leading token — the v0.71.32
    code-review CRITICAL).
    """
    if decoder_start_token_id is None or labels.shape[1] == 0:
        return labels
    if bool((labels[:, 0] == decoder_start_token_id).all().cpu().item()):
        return labels[:, 1:]
    return labels


class _SpeechSeq2SeqCollator:
    """Pad Whisper ``input_features`` + ``labels`` (labels pad -> -100).

    Standard HF speech-seq2seq collator: the feature extractor pads the log-mel
    features to a fixed shape; the tokenizer pads the label ids, and pad tokens
    are replaced with ``-100`` so they do not contribute to the loss. A leading
    ``decoder_start_token_id`` is stripped (the model re-prepends it).
    """

    def __init__(self, processor: Any, decoder_start_token_id: int | None = None) -> None:
        self.processor = processor
        self.decoder_start_token_id = decoder_start_token_id

    def __call__(self, features: list[dict]) -> dict:
        input_features = [
            {"input_features": f["input_features"]} for f in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )
        labels = _strip_decoder_start(labels, self.decoder_start_token_id)
        batch["labels"] = labels
        return batch
