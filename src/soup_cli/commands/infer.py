"""soup infer — batch inference on a list of prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

console = Console()


def _is_path_like(value: str) -> bool:
    """Heuristic: looks like a filesystem path rather than a HF repo id.

    HF repo ids are ``owner/name`` with no leading dot/slash and no Windows
    drive letter; anything else (``./foo``, ``/abs/path``, ``C:\\...``) is
    treated as a path so we surface a meaningful FileNotFoundError instead
    of attempting an HF download.
    """
    if not value:
        return True
    if value.startswith((".", "/", "\\", "~")):
        return True
    # Windows drive letter, e.g. "C:\..." or "C:/..."
    if len(value) >= 2 and value[1] == ":":
        return True
    return False


def _resolve_model_source(model: str) -> tuple[str, str]:
    """Return ``("local", path)`` or ``("hf", repo_id)`` for ``--model``.

    Falls through to HF when the local path doesn't exist *and* the value
    looks like a HF repo id (no leading ``./`` etc.). Raises
    :class:`FileNotFoundError` when the value looks like a path but doesn't
    exist locally — distinguishes "your file is missing" from "your HF id
    is wrong" so the error message is actionable.
    """
    candidate = Path(model)
    if candidate.exists():
        return "local", str(candidate)
    if _is_path_like(model):
        raise FileNotFoundError(f"Model path not found: {model}")
    # Looks like a HF repo id — let transformers handle the download.
    return "hf", model


def infer(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to model (LoRA adapter or full model)",
    ),
    input_file: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to input JSONL file (each line: {\"prompt\": \"...\"})",
    ),
    output_file: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Path to output JSONL file for results",
    ),
    base: Optional[str] = typer.Option(
        None,
        "--base",
        "-b",
        help="Base model for LoRA adapter (auto-detected if not set)",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        min=1,
        max=16384,
        help="Maximum tokens to generate per response (1-16384)",
    ),
    temperature: float = typer.Option(
        0.7,
        "--temperature",
        "-t",
        help="Sampling temperature (0 = greedy)",
    ),
    device: Optional[str] = typer.Option(
        None,
        "--device",
        help="Device: cuda, mps, cpu. Auto-detected if not set.",
    ),
    task: str = typer.Option(
        "text",
        "--task",
        help=(
            "Inference task: 'text' (default, chat generation) or 'asr' "
            "(Whisper transcription; input rows are {\"audio\": path[, "
            "\"text\": reference]})."
        ),
    ),
    asr_language: Optional[str] = typer.Option(
        None,
        "--asr-language",
        help="ASR decode language (overrides the training sidecar). --task asr only.",
    ),
    asr_task: Optional[str] = typer.Option(
        None,
        "--asr-task",
        help="ASR decode objective: transcribe | translate. --task asr only.",
    ),
    audio_dir: Optional[str] = typer.Option(
        None,
        "--audio-dir",
        help=(
            "Base directory audio paths in --input must stay under (--task asr; "
            "defaults to the --input file's directory). Traversal / UNC paths "
            "are rejected."
        ),
    ),
    trust_remote_code: bool = typer.Option(
        False,
        "--trust-remote-code",
        help=(
            "Allow loading models that ship custom Python via auto_map. "
            "Default deny (v0.36.0). Only enable if you trust the source."
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Source hub for the base model: hf (default) / modelscope / "
            "modelers. Non-HF hubs require the matching SDK (v0.53.10 #152)."
        ),
    ),
):
    """Run batch inference on a JSONL file of prompts."""
    # v0.53.10 #152 — pre-fetch base from a non-HF hub before any resolution.
    if hub and hub != "hf":
        from soup_cli.utils.hubs import apply_hub_to_cli_model

        try:
            model, base = apply_hub_to_cli_model(model, base, hub, console=console)
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=2) from exc
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(code=1) from exc

    from soup_cli.utils.paths import is_under_cwd

    # Validate input file
    input_path = Path(input_file)
    if not input_path.exists():
        console.print(f"[red]Input file not found: {input_path}[/]")
        raise typer.Exit(1)

    # v0.71.32 — ASR (Whisper) transcription branch. Diverts before the chat
    # model-resolution path; _infer_asr owns its own Whisper load + output.
    if task == "asr":
        _infer_asr(
            model=model,
            base=base,
            input_file=input_file,
            device=device,
            output_file=output_file,
            max_tokens=max_tokens,
            trust_remote_code=trust_remote_code,
            asr_language=asr_language,
            asr_task=asr_task,
            audio_dir=audio_dir,
        )
        return
    if task != "text":
        console.print(f"[red]Unknown --task {task!r}; expected 'text' or 'asr'.[/]")
        raise typer.Exit(2)

    # Resolve model: local path or HF repo id (auto-fallback, #N7).
    try:
        model_kind, model_ref = _resolve_model_source(model)
    except FileNotFoundError as exc:
        console.print(
            f"[red]{exc}[/]\n"
            "[dim]If you meant a HuggingFace repo, use the form "
            "'owner/repo-name' (no leading './').[/]"
        )
        raise typer.Exit(1) from exc
    model_path = Path(model_ref)
    if model_kind == "hf":
        console.print(
            f"[dim]Local path not found; treating {model_ref!r} as a HF repo id.[/]"
        )

    # Read prompts
    prompts = _read_prompts(input_path)
    if not prompts:
        console.print("[red]No prompts found in input file.[/]")
        console.print("[dim]Expected JSONL with {\"prompt\": \"...\"} or plain text lines.[/]")
        raise typer.Exit(1)

    # Detect device
    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    console.print(
        Panel(
            f"Model:    [bold]{model_path}[/]\n"
            f"Input:    [bold]{input_path}[/] ({len(prompts)} prompts)\n"
            f"Output:   [bold]{output_file}[/]\n"
            f"Device:   [bold]{device}[/]\n"
            f"Tokens:   [bold]{max_tokens}[/]\n"
            f"Temp:     [bold]{temperature}[/]",
            title="Batch Inference",
        )
    )

    # Load model — gate trust_remote_code via the v0.36.0 helper.
    console.print("[dim]Loading model...[/]")
    model_obj, tokenizer = _load_model(
        str(model_path), base, device, trust_remote_code,
    )
    console.print("[green]Model loaded.[/]\n")

    # Output path containment — defence-in-depth (project policy v0.20.0+).
    # Checked late, after model+inputs validate, so that pre-existing tests
    # asserting on "model not found" / "no prompts" errors keep working when
    # they pass an out-of-cwd `tmp_path`.
    if not is_under_cwd(output_file):
        console.print(
            "[red]--output must stay under the current working directory.[/]"
        )
        raise typer.Exit(1)

    # Run inference — stream results to disk as they are generated
    output_path = Path(output_file)
    total_tokens = 0
    num_results = 0
    start_time = time.time()

    with (
        open(output_path, "w", encoding="utf-8") as out_f,
        Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress,
    ):
        progress_task = progress.add_task("Generating...", total=len(prompts))

        for prompt_text in prompts:
            messages = [{"role": "user", "content": prompt_text}]
            response, token_count = _generate(
                model_obj, tokenizer, messages,
                max_tokens=max_tokens, temperature=temperature,
            )

            result = {
                "prompt": prompt_text,
                "response": response,
                "tokens_generated": token_count,
            }
            out_f.write(json.dumps(result, ensure_ascii=False) + "\n")
            out_f.flush()
            total_tokens += token_count
            num_results += 1
            progress.update(progress_task, advance=1)

    elapsed = time.time() - start_time
    tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0

    console.print(
        Panel(
            f"Prompts:       [bold]{num_results}[/]\n"
            f"Total tokens:  [bold]{total_tokens}[/]\n"
            f"Duration:      [bold]{elapsed:.1f}s[/]\n"
            f"Throughput:    [bold]{tokens_per_sec:.1f} tok/s[/]\n"
            f"Output:        [bold]{output_path}[/]",
            title="[bold green]Inference Complete![/]",
        )
    )


# v0.71.32 — test seam: when set to ``callable(audio_path) -> str`` it replaces
# the real Whisper transcriber, so the ASR path is unit-testable without a
# model download.
_ASR_TRANSCRIBER_OVERRIDE = None

# Cap on ASR batch rows — each row is an audio decode + a full generate(), far
# costlier than a text prompt, so an unbounded --input is a resource-exhaustion
# vector (mirrors the project's 10k custom-eval / 1e6 HF-download caps).
_MAX_ASR_ROWS: int = 100_000


def _read_asr_rows(path: Path) -> list[dict]:
    """Read ASR rows ``{"audio": path[, "text": reference]}`` from JSONL.

    Rows without a non-empty string ``audio`` are dropped (a warning is
    printed once). ``text``, when present, is the reference transcript used for
    WER reporting. Capped at ``_MAX_ASR_ROWS``.
    """
    rows: list[dict] = []
    dropped = 0
    capped = False
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                dropped += 1
                continue
            audio = obj.get("audio") if isinstance(obj, dict) else None
            if not isinstance(audio, str) or not audio.strip():
                dropped += 1
                continue
            rows.append(obj)
            if len(rows) >= _MAX_ASR_ROWS:
                capped = True
                break
    if dropped:
        console.print(
            f"[yellow]Skipped {dropped} row(s) with no 'audio' path.[/]"
        )
    if capped:
        console.print(
            f"[yellow]Capped at {_MAX_ASR_ROWS} rows; remaining input ignored.[/]"
        )
    return rows


def _resolve_asr_audio(audio: str, base_dir: Path) -> str:
    """Resolve a row's audio path against ``base_dir`` with containment.

    Rejects UNC / network paths and anything that resolves outside
    ``base_dir`` (realpath + commonpath) — the infer path is fed JSONL the
    operator may not have authored (the training path already enforces this
    via ``_validate_audio_files``). Raises ``ValueError`` on rejection.
    """
    from soup_cli.utils.paths import is_under

    if "\x00" in audio:
        raise ValueError("audio path must not contain null bytes")
    # UNC (\\host\share) / network (//host) paths trigger outbound SMB on
    # Windows — reject before any filesystem touch.
    if audio.startswith(("\\\\", "//")):
        raise ValueError("audio path must not be a UNC / network path")
    candidate = Path(audio)
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    if not is_under(candidate, base_dir):
        raise ValueError(
            f"audio path {Path(audio).name!r} must stay under the audio dir"
        )
    return str(candidate)


def _resolve_asr_gen_prefix(
    asr_language: Optional[str], asr_task: Optional[str], sidecar: dict
) -> dict:
    """Decode-prefix precedence: explicit CLI flags > training sidecar.

    Returns the ``language``/``task`` kwargs to pass to ``whisper.generate``
    (omitting either when unresolved so the model default applies).
    """
    gen_kwargs: dict = {}
    language = asr_language or sidecar.get("language")
    task = asr_task or sidecar.get("task")
    if language:
        gen_kwargs["language"] = language
    if task:
        gen_kwargs["task"] = task
    return gen_kwargs


def _build_asr_transcriber(
    model: str,
    base: Optional[str],
    device: Optional[str],
    max_tokens: int,
    trust_remote_code: bool,
    asr_language: Optional[str] = None,
    asr_task: Optional[str] = None,
) -> Callable[[str], str]:
    """Build a ``transcribe(audio_path) -> str`` closure over a Whisper model.

    Handles both a full fine-tuned Whisper directory and a PEFT/LoRA adapter
    dir (base resolved from ``--base`` or the adapter's
    ``base_model_name_or_path``). The decode language/task come from (in order)
    the explicit CLI flags, then the training ``asr_generation.json`` sidecar,
    then the model default.
    """
    from soup_cli.trainer.asr import _require_whisper_base, read_asr_sidecar
    from soup_cli.utils.tts_codec import load_audio_mono

    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    from soup_cli.utils.trust_remote import resolve_trust_remote_code

    # Detect a PEFT/LoRA adapter dir; resolve its base for the weight load.
    model_path = Path(model)
    adapter_cfg = model_path / "adapter_config.json"
    is_adapter = adapter_cfg.exists()
    base_ref = base
    if is_adapter and not base_ref:
        try:
            with open(adapter_cfg, encoding="utf-8") as fh:
                base_ref = json.load(fh).get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError):
            base_ref = None
        if not base_ref:
            console.print(
                f"[red]Cannot detect base model for adapter {model_path}; "
                "pass --base.[/]"
            )
            raise typer.Exit(1)

    # The base (adapter case) or the model itself carries the Whisper weights
    # AND the processor / arch identity.
    weights_ref = base_ref if is_adapter else model
    from soup_cli.utils.trust_remote import model_requires_trust_remote_code

    requires = model_requires_trust_remote_code(weights_ref) or False
    resolved_trust = resolve_trust_remote_code(
        weights_ref, requested=trust_remote_code, console=console,
        requires_remote_code=requires,
    )
    # Arch guard: reject a non-Whisper base before the (large) weight load.
    _require_whisper_base(weights_ref, resolved_trust)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    console.print(f"[dim]Loading Whisper model: {model}[/]")
    # Prefer the fine-tuned dir's processor (carries any resized vocab); fall
    # back to the base for an adapter-only dir.
    processor = WhisperProcessor.from_pretrained(
        model if not is_adapter else weights_ref, trust_remote_code=resolved_trust
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(
        weights_ref, trust_remote_code=resolved_trust
    )
    if is_adapter:
        from peft import PeftModel

        whisper = PeftModel.from_pretrained(whisper, model)
    whisper.to(device)
    whisper.eval()

    # Resolve decode prefix: explicit flags > training sidecar > model default.
    sidecar = read_asr_sidecar(model)
    gen_kwargs = _resolve_asr_gen_prefix(asr_language, asr_task, sidecar)
    gen_kwargs["max_new_tokens"] = max_tokens

    def transcribe(audio_path: str) -> str:
        import torch

        wave = load_audio_mono(audio_path, target_sr=16000)
        features = processor.feature_extractor(
            wave, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            generated = whisper.generate(features, **gen_kwargs)
        return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()

    return transcribe


def _infer_asr(
    *,
    model: str,
    base: Optional[str],
    input_file: str,
    device: Optional[str],
    output_file: str,
    max_tokens: int,
    trust_remote_code: bool,
    asr_language: Optional[str] = None,
    asr_task: Optional[str] = None,
    audio_dir: Optional[str] = None,
) -> None:
    """Transcribe an ASR JSONL input and (optionally) report WER/CER."""
    from soup_cli.utils.asr_metrics import cer, corpus_wer, wer
    from soup_cli.utils.paths import (
        atomic_write_text,
        enforce_under_cwd_and_no_symlink,
        is_under_cwd,
    )

    # Output containment + no-symlink (TOCTOU). Atomic write at the end.
    try:
        enforce_under_cwd_and_no_symlink(output_file, "--output")
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    # Audio containment base: --audio-dir (must be under cwd) or the input's dir.
    if audio_dir:
        if not is_under_cwd(audio_dir):
            console.print("[red]--audio-dir must stay under the cwd.[/]")
            raise typer.Exit(1)
        base_dir = Path(audio_dir)
    else:
        base_dir = Path(input_file).resolve().parent

    rows = _read_asr_rows(Path(input_file))
    if not rows:
        console.print("[red]No ASR rows found (need {\"audio\": path} JSONL).[/]")
        raise typer.Exit(1)

    if _ASR_TRANSCRIBER_OVERRIDE is not None:
        transcribe = _ASR_TRANSCRIBER_OVERRIDE
    else:
        try:
            transcribe = _build_asr_transcriber(
                model, base, device, max_tokens, trust_remote_code,
                asr_language=asr_language, asr_task=asr_task,
            )
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        except ValueError as exc:  # arch guard / bad base
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2) from exc

    refs: list[str] = []
    hyps: list[str] = []
    out_lines: list[str] = []
    skipped = 0
    for row in rows:
        audio = row["audio"]
        try:
            resolved = _resolve_asr_audio(audio, base_dir)
            hyp = transcribe(resolved)
        except (ValueError, OSError, ImportError) as exc:
            skipped += 1
            console.print(f"[yellow]Skipped {Path(str(audio)).name!r}: {exc}[/]")
            continue
        rec = {"audio": audio, "transcription": hyp}
        ref = row.get("text")
        if isinstance(ref, str):
            rec["reference"] = ref
            rec["wer"] = wer(ref, hyp)
            rec["cer"] = cer(ref, hyp)
            refs.append(ref)
            hyps.append(hyp)
        out_lines.append(json.dumps(rec, ensure_ascii=False))

    atomic_write_text("\n".join(out_lines) + ("\n" if out_lines else ""), output_file,
                      field="--output")

    summary = f"Transcribed [bold]{len(out_lines)}[/] clip(s) -> {output_file}"
    if skipped:
        summary += f"  ([yellow]{skipped} skipped[/])"
    if refs:
        summary += f"\nCorpus WER: [bold]{corpus_wer(refs, hyps):.3f}[/]"
    console.print(Panel(summary, title="[bold green]ASR Complete![/]"))


def _read_prompts(path: Path) -> list[str]:
    """Read prompts from a JSONL or plain text file."""
    prompts = []
    with open(path, encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            # Try JSONL
            try:
                obj = json.loads(line)
                if isinstance(obj, dict) and "prompt" in obj:
                    prompts.append(obj["prompt"])
                    continue
            except json.JSONDecodeError:
                pass
            # Plain text
            prompts.append(line)
    return prompts


def _load_model(
    model_path: str,
    base_model: Optional[str],
    device: str,
    trust_remote_code: bool = False,
) -> tuple:
    """Load a model and tokenizer (reuses diff.py pattern)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from soup_cli.utils.trust_remote import (
        model_requires_trust_remote_code,
        resolve_trust_remote_code,
    )

    path = Path(model_path)
    adapter_config_path = path / "adapter_config.json"
    is_adapter = adapter_config_path.exists()

    if is_adapter and not base_model:
        try:
            with open(adapter_config_path, encoding="utf-8") as f:
                config = json.load(f)
            base_model = config.get("base_model_name_or_path")
        except (json.JSONDecodeError, OSError):
            pass

    if is_adapter and not base_model:
        console.print(
            f"[red]Cannot detect base model for {path}. Use --base.[/]"
        )
        raise typer.Exit(1)

    probe_target = base_model or model_path
    requires = model_requires_trust_remote_code(model_path) or False
    trc = resolve_trust_remote_code(
        probe_target,
        requested=trust_remote_code,
        console=console,
        requires_remote_code=requires,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trc)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if is_adapter:
        from peft import PeftModel

        base_obj = AutoModelForCausalLM.from_pretrained(
            base_model,
            trust_remote_code=trc,
            device_map="auto",
            dtype=torch.float16,
        )
        model_obj = PeftModel.from_pretrained(base_obj, model_path)
    else:
        model_obj = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=trc,
            device_map="auto",
            dtype=torch.float16,
        )

    model_obj.eval()
    return model_obj, tokenizer


def _generate(
    model, tokenizer, messages, max_tokens=256, temperature=0.7,
) -> tuple[str, int]:
    """Generate a response from the model. Returns (text, token_count)."""
    import torch

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    else:
        parts = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "user":
                parts.append(f"User: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
        parts.append("Assistant:")
        text = "\n".join(parts)

    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = 0.9
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[0][input_ids.shape[1]:]
    token_count = new_tokens.shape[0]
    response_text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response_text, token_count
