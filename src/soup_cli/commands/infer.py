"""soup infer — batch inference on a list of prompts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

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
        task = progress.add_task("Generating...", total=len(prompts))

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
            progress.update(task, advance=1)

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


def _read_asr_rows(path: Path) -> list[dict]:
    """Read ASR rows ``{"audio": path[, "text": reference]}`` from JSONL.

    Rows without a non-empty string ``audio`` are dropped (a warning is
    printed once). ``text``, when present, is the reference transcript used for
    WER reporting.
    """
    rows: list[dict] = []
    dropped = 0
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
    if dropped:
        console.print(
            f"[yellow]Skipped {dropped} row(s) with no 'audio' path.[/]"
        )
    return rows


def _build_asr_transcriber(
    model: str,
    base: Optional[str],
    device: Optional[str],
    max_tokens: int,
    trust_remote_code: bool,
):
    """Build a ``transcribe(audio_path) -> str`` closure over a Whisper model."""
    from soup_cli.trainer.asr import _require_whisper_base
    from soup_cli.utils.tts_codec import load_audio_mono

    if not device:
        from soup_cli.utils.gpu import detect_device

        device, _ = detect_device()

    from soup_cli.utils.trust_remote import resolve_trust_remote_code

    source = base or model
    resolved_trust = resolve_trust_remote_code(
        source, requested=trust_remote_code, console=console,
    )
    # Arch guard: reject a non-Whisper base before the (large) weight load.
    _require_whisper_base(source, resolved_trust)

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    console.print(f"[dim]Loading Whisper model: {model}[/]")
    processor = WhisperProcessor.from_pretrained(
        source, trust_remote_code=resolved_trust
    )
    whisper = WhisperForConditionalGeneration.from_pretrained(
        model, trust_remote_code=resolved_trust
    )
    whisper.to(device)
    whisper.eval()

    def transcribe(audio_path: str) -> str:
        import torch

        wave = load_audio_mono(audio_path, target_sr=16000)
        features = processor.feature_extractor(
            wave, sampling_rate=16000, return_tensors="pt"
        ).input_features.to(device)
        with torch.no_grad():
            generated = whisper.generate(features, max_new_tokens=max_tokens)
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
) -> None:
    """Transcribe an ASR JSONL input and (optionally) report WER/CER."""
    from soup_cli.utils.asr_metrics import cer, corpus_wer, wer
    from soup_cli.utils.paths import is_under_cwd

    if not is_under_cwd(output_file):
        console.print(
            "[red]--output must stay under the current working directory.[/]"
        )
        raise typer.Exit(1)

    rows = _read_asr_rows(Path(input_file))
    if not rows:
        console.print("[red]No ASR rows found (need {\"audio\": path} JSONL).[/]")
        raise typer.Exit(1)

    if _ASR_TRANSCRIBER_OVERRIDE is not None:
        transcribe = _ASR_TRANSCRIBER_OVERRIDE
    else:
        try:
            transcribe = _build_asr_transcriber(
                model, base, device, max_tokens, trust_remote_code
            )
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        except ValueError as exc:  # arch guard / bad base
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2) from exc

    refs: list[str] = []
    hyps: list[str] = []
    output_path = Path(output_file)
    with open(output_path, "w", encoding="utf-8") as out_f:
        for row in rows:
            audio = row["audio"]
            hyp = transcribe(audio)
            rec = {"audio": audio, "transcription": hyp}
            ref = row.get("text")
            if isinstance(ref, str):
                rec["reference"] = ref
                rec["wer"] = wer(ref, hyp)
                rec["cer"] = cer(ref, hyp)
                refs.append(ref)
                hyps.append(hyp)
            out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    summary = f"Transcribed [bold]{len(rows)}[/] clip(s) -> {output_path}"
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
