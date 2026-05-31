"""soup push — upload a trained model to HuggingFace Hub."""

from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel

console = Console()

# Files that should exist in a valid LoRA adapter directory
ADAPTER_FILES = {"adapter_config.json", "adapter_model.safetensors"}
ADAPTER_FILES_ALT = {"adapter_config.json", "adapter_model.bin"}


def push(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="Path to the trained model / LoRA adapter directory",
    ),
    repo: str = typer.Option(
        ...,
        "--repo",
        "-r",
        help="HuggingFace repo ID, e.g. username/my-model",
    ),
    private: bool = typer.Option(
        False,
        "--private",
        help="Make the HuggingFace repo private",
    ),
    token: Optional[str] = typer.Option(
        None,
        "--token",
        "-t",
        help="[deprecated] Use HF_TOKEN env var instead. Falls back to cached login.",
        envvar="HF_TOKEN",
    ),
    commit_message: str = typer.Option(
        "Upload model trained with Soup CLI",
        "--message",
        help="Commit message for the upload",
    ),
    collection: Optional[str] = typer.Option(
        None,
        "--collection",
        help=(
            "Add the pushed repo to an existing HF Collection "
            "(slug: 'owner/title-hash')"
        ),
    ),
    hub: str = typer.Option(
        "hf",
        "--hub",
        help=(
            "Destination hub: hf (default) / modelscope / modelers. Non-HF "
            "hubs require the matching SDK and skip the HF-specific "
            "Collections / model-card auto-render path (v0.53.10 #152)."
        ),
    ),
):
    """Push a trained model to HuggingFace Hub (or alternate hub)."""
    # v0.53.10 #152 — validate hub at the CLI boundary; only HF is the
    # default. Non-HF hubs upload via :func:`utils.hubs.upload_repo` after
    # the standard model-dir validation completes.
    from soup_cli.utils.hubs import validate_hub_name

    try:
        hub_canonical = validate_hub_name(hub)
    except (TypeError, ValueError) as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(code=2) from exc

    from soup_cli.utils.paths import is_under_cwd

    model_path = Path(model)

    # --- Validate model directory ---
    if not model_path.exists():
        console.print(f"[red]Model path not found: {model_path}[/]")
        raise typer.Exit(1)

    if not model_path.is_dir():
        console.print(f"[red]Expected a directory, got a file: {model_path}[/]")
        raise typer.Exit(1)

    if not is_under_cwd(model_path):
        console.print(
            "[red]--model path must stay under the current working directory.[/]"
        )
        raise typer.Exit(1)

    # Deprecated --token flag: warn once if explicitly provided.
    if token is not None:
        console.print(
            "[yellow]Warning: --token is deprecated. Use HF_TOKEN env var or "
            "run 'huggingface-cli login'.[/]"
        )

    # Sanitise commit message: strip to first line, cap length so a crafted
    # multi-line message can't pollute HF commit history.
    commit_message = commit_message.splitlines()[0][:200] if commit_message else ""

    files_in_dir = {f.name for f in model_path.iterdir() if f.is_file()}
    is_adapter = ADAPTER_FILES.issubset(files_in_dir) or ADAPTER_FILES_ALT.issubset(files_in_dir)

    if not is_adapter and "config.json" not in files_in_dir:
        console.print(
            "[red]Directory does not look like a valid model or LoRA adapter.[/]\n"
            "Expected adapter_config.json (LoRA) or config.json (full model)."
        )
        raise typer.Exit(1)

    # --- Resolve HF token (env > cached login, see utils.hf.resolve_token) ---
    from soup_cli.utils.hf import resolve_endpoint, resolve_token, validate_repo_id

    try:
        validate_repo_id(repo)
    except ValueError as exc:
        console.print(f"[red]Invalid --repo:[/] {exc}")
        raise typer.Exit(1) from exc

    hf_token = resolve_token(explicit=token)
    if not hf_token:
        console.print(
            "[red]No HuggingFace token found.[/]\n"
            "Provide one via:\n"
            "  --token YOUR_TOKEN\n"
            "  HF_TOKEN=... env variable\n"
            "  huggingface-cli login"
        )
        raise typer.Exit(1)

    try:
        hf_endpoint = resolve_endpoint()
    except ValueError as exc:
        console.print(f"[red]HF_ENDPOINT invalid:[/] {exc}")
        raise typer.Exit(1) from exc

    # --- Show upload plan ---
    file_count = sum(1 for _ in model_path.rglob("*") if _.is_file())
    total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
    size_str = _format_size(total_size)

    console.print(
        Panel(
            f"Source:  [bold]{model_path}[/]\n"
            f"Repo:   [bold]{repo}[/]\n"
            f"Type:   [bold]{'LoRA adapter' if is_adapter else 'Full model'}[/]\n"
            f"Files:  [bold]{file_count}[/]\n"
            f"Size:   [bold]{size_str}[/]\n"
            f"Private: [bold]{private}[/]",
            title="Upload Plan",
        )
    )

    # --- Upload ---
    # v0.53.10 #152 — non-HF hubs route through utils.hubs.upload_repo
    # before we reach the HF-specific Collections / model-card auto-render
    # path. Each backend lazy-imports its own SDK; missing-dep surfaces
    # as ImportError with a pip-install advisory.
    if hub_canonical != "hf":
        from soup_cli.utils.hubs import upload_repo

        console.print(f"[dim]Uploading to hub={hub_canonical}...[/]")
        try:
            upload_repo(
                hub_canonical,
                repo,
                folder_path=str(model_path),
                commit_message=commit_message,
                token=hf_token,
            )
        except ImportError as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(1) from exc
        except (TypeError, ValueError) as exc:
            console.print(f"[red]{exc}[/]")
            raise typer.Exit(2) from exc
        console.print(
            f"[green]Pushed to {hub_canonical}/{repo}.[/]\n"
            "[dim]Note: HF-specific Collections + model card auto-render "
            "are HF-only; install via the HF flow for those features.[/]"
        )
        return

    console.print("[dim]Uploading to HuggingFace Hub...[/]")

    from soup_cli.utils.hf import get_hf_api

    try:
        api = get_hf_api(token=hf_token, endpoint=hf_endpoint)
    except ImportError as exc:
        console.print(
            "[red]huggingface-hub not installed.[/]\n"
            "Run: [bold]pip install huggingface-hub[/]"
        )
        raise typer.Exit(1) from exc

    try:
        # Create repo if it doesn't exist
        api.create_repo(repo_id=repo, private=private, exist_ok=True)

        # Upload the entire directory
        api.upload_folder(
            folder_path=str(model_path),
            repo_id=repo,
            commit_message=commit_message,
        )

        # Generate and upload model card if not present (v2 — includes
        # training config and optional eval scorecard)
        readme_path = model_path / "README.md"
        if not readme_path.exists():
            model_card = generate_model_card_v2(
                model_path, repo_id=repo, is_adapter=is_adapter,
            )
            api.upload_file(
                path_or_fileobj=model_card.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo,
                commit_message="Add model card (generated by Soup CLI)",
            )
    except Exception as exc:
        console.print(f"[red]Upload failed: {exc}[/]")
        raise typer.Exit(1) from exc

    # --- Optional: add to Collection ---
    if collection:
        from soup_cli.utils.hf import (
            add_to_collection,
            validate_collection_slug,
        )
        from soup_cli.utils.hf import (
            resolve_endpoint as _resolve_endpoint,
        )

        try:
            validate_collection_slug(collection)
        except ValueError as exc:
            console.print(f"[red]Invalid --collection slug:[/] {exc}")
            raise typer.Exit(1) from exc

        try:
            endpoint = _resolve_endpoint()
        except ValueError as exc:
            console.print(f"[red]Collection: {exc}[/]")
            raise typer.Exit(1) from exc
        try:
            add_to_collection(
                collection_slug=collection,
                repo_id=repo,
                token=hf_token,
                endpoint=endpoint,
                item_type="model",
            )
            console.print(f"[green]Added to collection:[/] {collection}")
        except Exception as exc:
            console.print(f"[yellow]Could not add to collection:[/] {exc}")

    repo_url = f"https://huggingface.co/{repo}"
    console.print(
        Panel(
            f"Repo: [bold blue]{repo_url}[/]\n\n"
            f"Use it:\n"
            f"  [bold]soup chat --model {repo}[/]\n"
            f"  [bold]from peft import PeftModel[/]",
            title="[bold green]Upload Complete![/]",
        )
    )


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _generate_model_card(model_path: Path, repo_id: str, is_adapter: bool) -> str:
    """Generate a basic model card README (legacy, kept for backward compat)."""
    return generate_model_card_v2(model_path, repo_id=repo_id, is_adapter=is_adapter)


def _load_adapter_config(model_path: Path) -> dict:
    """Read ``adapter_config.json`` if present, return {} on any error."""
    config_path = model_path / "adapter_config.json"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except (json.JSONDecodeError, OSError):
        pass
    return {}


def _load_training_config(model_path: Path) -> dict:
    """Read sidecar ``training_config.yaml`` written by Soup training runs."""
    for name in ("training_config.yaml", "soup.yaml"):
        path = model_path / name
        if not path.exists():
            continue
        try:
            import yaml
        except ImportError:
            return {}
        try:
            with open(path, encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
            if isinstance(data, dict):
                return data
        except (yaml.YAMLError, OSError):
            continue
    return {}


_UNSAFE_MD_CHARS = re.compile(r"[\|\[\]\(\)!\n\r\t<>]")


def _safe_md_cell(value: str) -> str:
    """Neutralise Markdown-active chars so ``value`` cannot inject table rows,
    links, images, or raw HTML when rendered on HF Hub."""
    return _UNSAFE_MD_CHARS.sub(" ", str(value)).strip()


def _render_eval_scorecard(eval_scorecard: Optional[dict]) -> str:
    if not eval_scorecard or not isinstance(eval_scorecard, dict):
        return ""
    lines = ["## Evaluation", "", "| Task | Score |", "| --- | --- |"]
    for task, score in eval_scorecard.items():
        try:
            numeric = float(score)
            formatted = f"{numeric:.3f}"
        except (TypeError, ValueError):
            formatted = _safe_md_cell(score)
        safe_task = _safe_md_cell(task) or "task"
        lines.append(f"| {safe_task} | {formatted} |")
    lines.append("")
    return "\n".join(lines)


def _render_training_section(training_cfg: dict) -> str:
    if not training_cfg:
        return ""
    task = training_cfg.get("task") or "sft"
    training = training_cfg.get("training", {}) or {}
    base = training_cfg.get("base") or ""
    lines = ["## Training", "", f"- **Task:** {task}"]
    if base:
        lines.append(f"- **Base model:** `{base}`")
    for key in ("epochs", "lr", "batch_size", "optimizer", "scheduler"):
        if key in training:
            lines.append(f"- **{key}:** {training[key]}")
    recipe = training_cfg.get("recipe")
    if recipe:
        lines.append(f"- **Recipe:** `{recipe}`")
    lines.append("")
    return "\n".join(lines)


def generate_model_card_v2(
    model_path: Path,
    repo_id: str,
    is_adapter: Optional[bool] = None,
    eval_scorecard: Optional[dict] = None,
    data_lineage: Optional[str] = None,
) -> str:
    """Model card v2 — enriched with eval scorecard, training config, lineage.

    This is the generator invoked by both ``soup push`` (manual upload) and
    the auto-push callback. When the training run wrote a sidecar
    ``training_config.yaml`` next to the adapter, we surface task / base /
    learning rate / optimizer in the card. When the caller passes a
    ``eval_scorecard`` dict, it is rendered as a markdown table.
    """
    adapter_config = _load_adapter_config(model_path)
    detected_adapter = bool(adapter_config) or (model_path / "adapter_config.json").exists()
    if is_adapter is None:
        is_adapter = detected_adapter

    adapter_info = ""
    if is_adapter and adapter_config:
        base = adapter_config.get("base_model_name_or_path", "unknown")
        lora_r = adapter_config.get("r", "?")
        lora_alpha = adapter_config.get("lora_alpha", "?")
        adapter_info = (
            f"- **Base model:** `{base}`\n"
            f"- **LoRA rank:** {lora_r}\n"
            f"- **LoRA alpha:** {lora_alpha}\n"
        )

    training_cfg = _load_training_config(model_path)
    training_section = _render_training_section(training_cfg)
    eval_section = _render_eval_scorecard(eval_scorecard)
    lineage_section = ""
    if data_lineage:
        # HTML-escape to block script / javascript: / img-onerror injection
        # on the HF Hub README viewer. Markdown chars remain visible but
        # inert.
        lineage_section = (
            f"## Data Lineage\n\n{html.escape(str(data_lineage))}\n"
        )

    model_name = repo_id.split("/")[-1] if "/" in repo_id else repo_id
    tags_block = "\n".join(
        [
            "tags:",
            "  - soup-cli",
            "  - fine-tuned",
            "  - lora" if is_adapter else "  - full-model",
        ]
    )
    library = "peft" if is_adapter else "transformers"

    if adapter_info:
        details_block = adapter_info
    else:
        details_block = "This is a fine-tuned language model."

    usage_block = (
        "```python\n"
        "from peft import PeftModel\n"
        "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
        'model = AutoModelForCausalLM.from_pretrained("BASE_MODEL")\n'
        f'model = PeftModel.from_pretrained(model, "{repo_id}")\n'
        f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
        "```\n"
        if is_adapter
        else (
            "```python\n"
            "from transformers import AutoModelForCausalLM, AutoTokenizer\n\n"
            f'model = AutoModelForCausalLM.from_pretrained("{repo_id}")\n'
            f'tokenizer = AutoTokenizer.from_pretrained("{repo_id}")\n'
            "```\n"
        )
    )

    sections = [
        "---",
        tags_block,
        f"library_name: {library}",
        "---",
        "",
        f"# {model_name}",
        "",
        "Fine-tuned model uploaded with [Soup CLI](https://github.com/MakazhanAlpamys/Soup).",
        "",
        "## Model Details",
        "",
        details_block,
    ]
    if training_section:
        sections.append(training_section)
    if eval_section:
        sections.append(eval_section)
    if lineage_section:
        sections.append(lineage_section)
    tail = [
        "## Usage",
        "",
        usage_block,
        "Or with Soup CLI:",
        "",
        "```bash",
        f"soup chat --model {repo_id}",
        "```",
        "",
    ]
    if not training_section:
        tail.extend(
            [
                "## Training",
                "",
                "Trained using [Soup CLI]"
                "(https://github.com/MakazhanAlpamys/Soup) "
                "— fine-tune LLMs in one command.",
                "",
            ]
        )
    sections.extend(tail)
    return "\n".join(sections)
