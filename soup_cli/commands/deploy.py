"""soup deploy — deploy models to inference runtimes (Ollama, HF Spaces)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

app = typer.Typer(no_args_is_help=True)


# ---------------------------------------------------------------------------
# HF Space templates (Part F of v0.29.0)
# ---------------------------------------------------------------------------

_GRADIO_APP_PY = '''"""Soup CLI-generated Gradio Chat Space."""

import os
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "{MODEL_REPO}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")


def respond(message, history):
    messages = []
    for user, assistant in history:
        messages.append({{"role": "user", "content": user}})
        messages.append({{"role": "assistant", "content": assistant}})
    messages.append({{"role": "user", "content": message}})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs, max_new_tokens=512, do_sample=True,
        temperature=0.7, top_p=0.9,
    )
    reply = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
    )
    return reply


chat = gr.ChatInterface(respond, title="Soup CLI Fine-tuned Chat")
chat.launch()
'''

_GRADIO_REQS = """gradio>=4.0.0
transformers>=4.40.0
torch>=2.1.0
accelerate>=0.27.0
"""

_GRADIO_README = """---
title: Soup Chat
emoji: 🍲
colorFrom: purple
colorTo: cyan
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
---

# Soup Chat

Generated with [Soup CLI](https://github.com/MakazhanAlpamys/Soup) — model: `{MODEL_REPO}`
"""


_STREAMLIT_APP_PY = '''"""Soup CLI-generated Streamlit Chat Space."""

import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "{MODEL_REPO}"

st.set_page_config(page_title="Soup Chat", page_icon="🍲")
st.title("Soup Chat")


@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="auto")
    return tokenizer, model


tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

prompt = st.chat_input("Ask me anything")
if prompt:
    st.session_state["messages"].append({{"role": "user", "content": prompt}})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        messages = [
            {{"role": m["role"], "content": m["content"]}}
            for m in st.session_state["messages"]
        ]
        chat_prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs, max_new_tokens=512, do_sample=True,
            temperature=0.7, top_p=0.9,
        )
        reply = tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True,
        )
        st.markdown(reply)
        st.session_state["messages"].append(
            {{"role": "assistant", "content": reply}}
        )
'''

_STREAMLIT_REQS = """streamlit>=1.30.0
transformers>=4.40.0
torch>=2.1.0
accelerate>=0.27.0
"""

_STREAMLIT_README = """---
title: Soup Chat
emoji: 🍲
colorFrom: purple
colorTo: cyan
sdk: streamlit
sdk_version: 1.30.0
app_file: app.py
pinned: false
---

# Soup Chat

Generated with [Soup CLI](https://github.com/MakazhanAlpamys/Soup) — model: `{MODEL_REPO}`
"""


HF_SPACE_TEMPLATES = {
    "gradio-chat": {
        "sdk": "gradio",
        "app.py": _GRADIO_APP_PY,
        "requirements.txt": _GRADIO_REQS,
        "README.md": _GRADIO_README,
    },
    "streamlit-chat": {
        "sdk": "streamlit",
        "app.py": _STREAMLIT_APP_PY,
        "requirements.txt": _STREAMLIT_REQS,
        "README.md": _STREAMLIT_README,
    },
}


def render_space_template(template: str, model_repo: str) -> dict[str, str]:
    """Render the Space template files with ``model_repo`` substituted.

    Raises ``ValueError`` when the template is unknown or the model repo id
    fails validation — injection-proof since repo ids are already a
    restrictive alphanumeric subset after validation.
    """
    from soup_cli.utils.hf import validate_repo_id

    validate_repo_id(model_repo)
    if template not in HF_SPACE_TEMPLATES:
        raise ValueError(
            f"Unknown template: {template!r}. "
            f"Available: {', '.join(HF_SPACE_TEMPLATES.keys())}"
        )
    spec = HF_SPACE_TEMPLATES[template]
    return {
        "app.py": spec["app.py"].replace("{MODEL_REPO}", model_repo),
        "requirements.txt": spec["requirements.txt"],
        "README.md": spec["README.md"].replace("{MODEL_REPO}", model_repo),
    }


@app.command()
def ollama(
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Path to GGUF model file",
    ),
    name: Optional[str] = typer.Option(
        None,
        "--name",
        "-n",
        help="Ollama model name (e.g. soup-my-model)",
    ),
    system: Optional[str] = typer.Option(
        None,
        "--system",
        "-s",
        help="System prompt for the model",
    ),
    template: str = typer.Option(
        "auto",
        "--template",
        "-t",
        help="Chat template: auto, chatml, llama, mistral, vicuna, zephyr",
    ),
    parameter: Optional[List[str]] = typer.Option(
        None,
        "--parameter",
        "-p",
        help="Ollama parameter (repeatable): temperature=0.7, top_p=0.9, etc.",
    ),
    list_models: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List Soup-deployed models in Ollama",
    ),
    remove: Optional[str] = typer.Option(
        None,
        "--remove",
        "-r",
        help="Remove a model from Ollama by name",
    ),
    yes: bool = typer.Option(
        False,
        "--yes",
        "-y",
        help="Skip confirmation prompt",
    ),
):
    """Deploy a GGUF model to local Ollama instance."""
    from soup_cli.utils.ollama import (
        OLLAMA_TEMPLATES,
        create_modelfile,
        deploy_to_ollama,
        detect_ollama,
        list_soup_models,
        remove_model,
        validate_gguf_path,
        validate_model_name,
    )

    # --- List mode ---
    if list_models:
        version = detect_ollama()
        if not version:
            console.print("[red]Ollama not found.[/] Install from https://ollama.com")
            raise typer.Exit(1)

        models = list_soup_models()
        if not models:
            console.print("[yellow]No Soup-deployed models found in Ollama.[/]")
            console.print("[dim]Deploy a model with: soup deploy ollama --model <gguf>[/]")
            raise typer.Exit(0)

        table = Table(title="Soup Models in Ollama")
        table.add_column("Name", style="bold cyan")
        table.add_column("Size", style="green")
        for entry in models:
            table.add_row(entry["name"], entry["size"])
        console.print(table)
        raise typer.Exit(0)

    # --- Remove mode ---
    if remove:
        valid_name, name_err = validate_model_name(remove)
        if not valid_name:
            console.print(f"[red]Invalid model name:[/] {name_err}")
            raise typer.Exit(1)

        version = detect_ollama()
        if not version:
            console.print("[red]Ollama not found.[/] Install from https://ollama.com")
            raise typer.Exit(1)

        if not yes:
            confirm = typer.confirm(f"Remove model '{remove}' from Ollama?")
            if not confirm:
                raise typer.Exit(0)

        success, message = remove_model(remove)
        if success:
            console.print(f"[green]{message}[/]")
        else:
            console.print(f"[red]{message}[/]")
            raise typer.Exit(1)
        raise typer.Exit(0)

    # --- Deploy mode: require --model and --name ---
    if not model:
        console.print("[red]--model is required for deploy.[/]")
        console.print("[dim]Usage: soup deploy ollama --model <gguf> --name <name>[/]")
        raise typer.Exit(1)

    if not name:
        console.print("[red]--name is required for deploy.[/]")
        console.print("[dim]Usage: soup deploy ollama --model <gguf> --name <name>[/]")
        raise typer.Exit(1)

    # Validate model name
    valid_name, name_err = validate_model_name(name)
    if not valid_name:
        console.print(f"[red]Invalid model name:[/] {name_err}")
        raise typer.Exit(1)

    # Validate GGUF path
    gguf_path = Path(model)
    valid_path, path_err = validate_gguf_path(gguf_path)
    if not valid_path:
        console.print(f"[red]{path_err}[/]")
        raise typer.Exit(1)

    # Check Ollama is installed
    version = detect_ollama()
    if not version:
        console.print(
            "[red]Ollama not found.[/]\n"
            "Install from: [bold]https://ollama.com[/]"
        )
        raise typer.Exit(1)

    # Resolve template
    resolved_template = None
    if template == "auto":
        # Try to infer from soup.yaml in cwd
        resolved_template = _auto_detect_template()
        if not resolved_template:
            resolved_template = "chatml"  # Default fallback
    elif template in OLLAMA_TEMPLATES:
        resolved_template = template
    else:
        console.print(
            f"[red]Unknown template: {template}[/]\n"
            f"Available: auto, {', '.join(OLLAMA_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    # Parse parameters
    params = {}
    if parameter:
        for param_str in parameter:
            if "=" not in param_str:
                console.print(f"[red]Invalid parameter format: {param_str}[/]")
                console.print("[dim]Expected format: key=value (e.g. temperature=0.7)[/]")
                raise typer.Exit(1)
            key, value = param_str.split("=", 1)
            params[key.strip()] = value.strip()

    # Show deploy plan
    console.print(
        Panel(
            f"Model:    [bold]{name}[/]\n"
            f"GGUF:     [bold]{gguf_path}[/]\n"
            f"Template: [bold]{resolved_template}[/]"
            + (f"\nSystem:   [bold]{system}[/]" if system else "")
            + (f"\nParams:   [bold]{params}[/]" if params else ""),
            title="Deploy to Ollama",
        )
    )

    # Confirmation — warn that this overwrites an existing model
    if not yes:
        console.print(
            "[yellow]Warning:[/] This will overwrite any existing Ollama model "
            f"named '{name}'."
        )
        confirm = typer.confirm("Proceed?")
        if not confirm:
            raise typer.Exit(0)

    # Generate Modelfile
    console.print(f"[green]\u2713[/] Ollama v{version} detected")
    try:
        modelfile = create_modelfile(
            gguf_path=gguf_path,
            template=resolved_template,
            system_prompt=system,
            parameters=params,
        )
    except ValueError as exc:
        console.print(f"[red]Invalid parameter:[/] {exc}")
        raise typer.Exit(1)
    console.print("[green]\u2713[/] Modelfile generated")

    # Deploy
    console.print("[dim]Creating model in Ollama...[/]")
    success, message = deploy_to_ollama(name, modelfile)
    if not success:
        console.print(f"[red]Deploy failed:[/] {message}")
        raise typer.Exit(1)

    console.print(f"[green]\u2713[/] Model created: [bold]{name}[/]")
    console.print(
        Panel(
            f"Run: [bold]ollama run {name}[/]",
            title="[bold green]Deploy Complete![/]",
        )
    )


@app.command(name="hf-space")
def hf_space(
    model: str = typer.Option(
        ...,
        "--model",
        "-m",
        help="HuggingFace model repo id to wrap in the Space (e.g. user/my-model)",
    ),
    space: str = typer.Option(
        ...,
        "--space",
        "-s",
        help="HuggingFace Space repo id to create (e.g. user/my-space)",
    ),
    template: str = typer.Option(
        "gradio-chat",
        "--template",
        "-t",
        help=f"Space template: {', '.join(HF_SPACE_TEMPLATES.keys())}",
    ),
    template_dir: Optional[str] = typer.Option(
        None,
        "--template-dir",
        help=(
            "Custom template directory (overrides --template). "
            "Must contain app.py + README.md, optionally requirements.txt. "
            "Use {MODEL_REPO} placeholder for substitution."
        ),
    ),
    private: bool = typer.Option(
        False, "--private", help="Create the Space as private",
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip confirmation",
    ),
):
    """Create a HuggingFace Space wrapping a fine-tuned model.

    Uploads an app.py, requirements.txt, and README.md rendered from the
    chosen template. Supports gradio-chat and streamlit-chat.
    """
    from soup_cli.utils.hf import (
        get_hf_api,
        resolve_endpoint,
        resolve_token,
        validate_repo_id,
    )

    # --- Validate space repo id up-front; model is validated by
    # render_space_template which is the authoritative entry point. ---
    try:
        validate_repo_id(space)
    except ValueError as exc:
        console.print(f"[red]Invalid --space repo id:[/] {exc}")
        raise typer.Exit(1) from exc
    if template_dir is None and template not in HF_SPACE_TEMPLATES:
        console.print(
            f"[red]Unknown template: {template}[/]\n"
            f"Available: {', '.join(HF_SPACE_TEMPLATES.keys())}"
        )
        raise typer.Exit(1)

    # --- Resolve credentials ---
    token = resolve_token()
    if token is None:
        console.print(
            "[red]No HuggingFace token found.[/]\n"
            "Set HF_TOKEN env var or run: [bold]huggingface-cli login[/]"
        )
        raise typer.Exit(1)

    try:
        endpoint = resolve_endpoint()
    except ValueError as exc:
        console.print(f"[red]HF_ENDPOINT invalid:[/] {exc}")
        raise typer.Exit(1) from exc

    # --- Render template files ---
    try:
        if template_dir is not None:
            from soup_cli.utils.hf_space import render_custom_template_dir
            files = render_custom_template_dir(template_dir, model_repo=model)
            # Custom templates default to gradio SDK unless requirements
            # imply otherwise; we record gradio for create_repo space_sdk.
            sdk = "gradio"
        else:
            files = render_space_template(template, model_repo=model)
            sdk = HF_SPACE_TEMPLATES[template]["sdk"]
    except ValueError as exc:
        console.print(f"[red]Template render failed:[/] {exc}")
        raise typer.Exit(1) from exc
    except FileNotFoundError as exc:
        console.print(f"[red]Template directory error:[/] {exc}")
        raise typer.Exit(1) from exc

    template_label = template_dir if template_dir is not None else template
    console.print(
        Panel(
            f"Space:    [bold]{space}[/]\n"
            f"Model:    [bold]{model}[/]\n"
            f"Template: [bold]{template_label}[/]\n"
            f"Private:  [bold]{private}[/]",
            title="Deploy HuggingFace Space",
        )
    )
    if not yes:
        confirm = typer.confirm("Create Space and upload files?", default=True)
        if not confirm:
            raise typer.Exit(0)

    # --- Create repo + upload ---
    try:
        api = get_hf_api(token=token, endpoint=endpoint)
    except ImportError as exc:
        console.print(f"[red]{exc}[/]")
        raise typer.Exit(1) from exc

    try:
        api.create_repo(
            repo_id=space, repo_type="space",
            space_sdk=sdk,
            private=private, exist_ok=True,
        )
        for in_repo_name, content in files.items():
            api.upload_file(
                path_or_fileobj=content.encode("utf-8"),
                path_in_repo=in_repo_name,
                repo_id=space,
                repo_type="space",
                commit_message=f"Soup CLI: add {in_repo_name}",
            )
    except Exception as exc:
        console.print(f"[red]Space deploy failed:[/] {exc}")
        raise typer.Exit(1) from exc

    console.print(
        Panel(
            f"Space:  [bold blue]https://huggingface.co/spaces/{space}[/]\n"
            f"Model:  [bold]{model}[/]",
            title="[bold green]Space Deployed![/]",
        )
    )


@app.command(name="autopilot")
def autopilot(
    target: Optional[str] = typer.Option(
        None,
        "--target",
        "-t",
        help="Profile name (e.g. mac-m3, rtx-4090-24gb, ollama-local).",
    ),
    base: str = typer.Option(
        "meta-llama/Llama-3.2-1B",
        "--base",
        "-b",
        help="Base model HF repo id or local path to embed in the recipe.",
    ),
    recipe_out: str = typer.Option(
        "deploy_autopilot.yaml",
        "--recipe-out",
        help="Where to write the rendered soup.yaml recipe (under cwd).",
    ),
    script_out: str = typer.Option(
        "deploy_autopilot.sh",
        "--script-out",
        help="Where to write the planned deploy shell script (under cwd).",
    ),
    output_dir: str = typer.Option(
        "./output",
        "--output-dir",
        help="Recipe's training output directory.",
    ),
    list_targets: bool = typer.Option(
        False,
        "--list",
        "-l",
        help="List all known deploy profiles.",
    ),
):
    """Pick PEFT + quant + spec-decoding combo for a hardware target.

    Writes a ready-to-train ``soup.yaml`` recipe and a planned deploy
    shell script. Live Quant-Lobotomy measurement deferred to v0.46.1.
    """
    from soup_cli.utils.deploy_autopilot import (
        get_profile,
        list_profiles,
        write_deploy_script,
        write_recipe,
    )

    if list_targets:
        table = Table(title="Deploy Autopilot Profiles")
        table.add_column("Name", style="bold cyan")
        table.add_column("Runtime", style="magenta")
        table.add_column("Quant", style="green")
        table.add_column("PEFT", style="yellow")
        table.add_column("Description")
        for profile in list_profiles().values():
            table.add_row(
                profile.name,
                profile.runtime,
                profile.quant,
                profile.peft,
                profile.description,
            )
        console.print(table)
        raise typer.Exit(0)

    if not target:
        console.print("[red]--target is required (or use --list).[/]")
        raise typer.Exit(2)

    try:
        profile = get_profile(target)
    except (KeyError, ValueError, TypeError) as exc:
        from rich.markup import escape

        console.print(f"[red]Unknown profile:[/] {escape(str(exc))}")
        console.print(
            "[dim]Run [bold]soup deploy autopilot --list[/] to see options.[/]"
        )
        raise typer.Exit(2) from exc

    try:
        recipe_path = write_recipe(
            profile, base=base, output_dir=output_dir, recipe_path=recipe_out
        )
        script_path = write_deploy_script(
            profile, model_path=output_dir, script_path=script_out
        )
    except (ValueError, TypeError) as exc:
        from rich.markup import escape

        console.print(f"[red]Autopilot failed:[/] {escape(str(exc))}")
        raise typer.Exit(1) from exc

    from rich.markup import escape as _escape

    console.print(
        Panel(
            f"Target:   [bold]{_escape(profile.name)}[/]\n"
            f"Runtime:  [bold]{_escape(profile.runtime)}[/]\n"
            f"Quant:    [bold]{_escape(profile.quant)}[/]\n"
            f"PEFT:     [bold]{_escape(profile.peft)}[/]\n"
            f"Spec dec: [bold]{profile.spec_decoding}[/]\n"
            f"Recipe:   [bold]{_escape(recipe_path)}[/]\n"
            f"Script:   [bold]{_escape(script_path)}[/]",
            title="[bold green]Deploy Autopilot[/]",
        )
    )
    if profile.notes:
        console.print(f"[dim]Notes: {_escape(profile.notes)}[/]")
    console.print(
        "[yellow]Note:[/] Live Quant-Lobotomy auto-measure deferred to v0.46.1; "
        "this release writes the canonical combo + recipe."
    )


def _auto_detect_template() -> Optional[str]:
    """Try to infer chat template from soup.yaml in cwd."""
    from soup_cli.utils.ollama import infer_chat_template

    config_path = Path("soup.yaml")
    if not config_path.exists():
        return None

    try:
        import yaml

        with open(config_path, encoding="utf-8") as fh:
            config = yaml.safe_load(fh)
        if not isinstance(config, dict):
            return None
        data_section = config.get("data", {})
        if isinstance(data_section, dict):
            fmt = data_section.get("format")
            return infer_chat_template(fmt)
    except (yaml.YAMLError, OSError, KeyError, ImportError):
        return None
    return None
