"""soup init — interactive project setup wizard."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soup_cli.templates import list_templates, load_template

console = Console()


def _template_help_string() -> str:
    """v0.40.1 Part D / H4 — generate help dynamically from the registry so
    the list never drifts away from `templates/manifest.json`.
    """
    return "Template: " + ", ".join(list_templates())


def init(
    template: str = typer.Option(
        None,
        "--template",
        "-t",
        help=_template_help_string(),
    ),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output config file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config without prompting (v0.40.1 / M2).",
    ),
    wizard: bool = typer.Option(
        False,
        "--wizard",
        "-w",
        help="Run smart interactive setup wizard with hardware autopilot.",
    ),
):
    """Create a new soup.yaml config interactively or from a template."""
    output_path = Path(output)

    if wizard:
        _smart_wizard(output_path=output_path, force=force)
        return

    if output_path.exists() and not force:
        overwrite = typer.confirm(f"{output_path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit()

    if template:
        config_text = load_template(template)
        if config_text is None:
            console.print(f"[red]Unknown template: {template}[/]")
            console.print(f"Available: {', '.join(list_templates())}")
            raise typer.Exit(1)
        console.print(f"[green]Using template:[/] {template}")
    else:
        config_text = _interactive_wizard()

    output_path.write_text(config_text, encoding="utf-8")
    console.print(
        Panel(
            f"[bold green]Config saved to {output_path}[/]\n\n"
            f"Next step: [bold]soup train --config {output_path}[/]",
            title="Ready!",
        )
    )


def wizard(
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output config file path",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing config without prompting.",
    ),
):
    """Smart interactive setup wizard that probes hardware and builds config."""
    output_path = Path(output)
    _smart_wizard(output_path=output_path, force=force)


def _smart_wizard(output_path: Path, force: bool = False) -> None:
    """Guided wizard that probes hardware, data, and suggests optimal hyperparameters."""
    import yaml
    from rich.table import Table

    from soup_cli.commands.autopilot import build_soup_config, decisions
    from soup_cli.config.schema import SoupConfig
    from soup_cli.utils.data import analyze_dataset
    from soup_cli.utils.hardware import analyze_hardware

    if output_path.exists() and not force:
        overwrite = typer.confirm(f"{output_path} already exists. Overwrite?")
        if not overwrite:
            raise typer.Exit()

    console.print(
        Panel.fit(
            "[bold cyan]🍲 Soup Smart Setup Wizard[/]\n"
            "[dim]Probing hardware and tailoring your fine-tuning recipe...[/]",
            border_style="cyan",
        )
    )

    # 1. Hardware probe
    hw = analyze_hardware()
    gpu_label = (
        f"[green]{hw.gpu_name} ({hw.total_vram_gb:.1f} GB VRAM)[/]"
        if hw.has_gpu
        else "[yellow]CPU only[/]"
    )
    console.print(f"Detected hardware: {gpu_label} | RAM: {hw.ram_gb:.1f} GB")

    # 2. Interactive Prompts
    base_model = Prompt.ask(
        "Base model (Hugging Face ID or local path)",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )

    goal_options = (
        "1) Chat / Instruction following (SFT)\n"
        "2) Reasoning / Math / Logic (GRPO)\n"
        "3) Preference alignment (DPO)\n"
        "4) Domain adaptation / Continued pre-training"
    )
    console.print(Panel(goal_options, title="Fine-tuning Goal", border_style="blue"))
    goal_choice = Prompt.ask("Select goal [1-4]", choices=["1", "2", "3", "4"], default="1")

    goal_map = {
        "1": ("chat", "sft"),
        "2": ("reasoning", "grpo"),
        "3": ("alignment", "dpo"),
        "4": ("general", "pretrain"),
    }
    goal_name, target_task = goal_map[goal_choice]

    data_path = Prompt.ask(
        "Training data path (JSONL, CSV, or parquet)",
        default="./data/train.jsonl",
    )

    # 3. Dataset inspection (if file exists)
    data_file = Path(data_path)
    if target_task == "sft":
        data_format = "chatml"
    elif target_task == "dpo":
        data_format = "dpo"
    else:
        data_format = "plaintext"
    if data_file.exists():
        try:
            profile = analyze_dataset(str(data_file))
            if profile.detected_format:
                data_format = profile.detected_format
            console.print(
                f"[green]✓ Dataset profiled:[/] {profile.num_rows} rows, "
                f"format: [bold]{data_format}[/], avg tokens: {profile.avg_tokens:.0f}"
            )
        except Exception:
            console.print(f"[dim]Note: Could not inspect {data_file}, using default format.[/]")
    else:
        console.print(f"[dim]Note: {data_file} not found yet; creating config for it.[/]")

    # 4. Compute recipe via autopilot decision engine
    decision = decisions(
        model=base_model,
        data_path=str(data_file),
        goal=goal_name,
        target_task=target_task,
        hw=hw,
    )
    soup_dict = build_soup_config(decision)
    # Ensure dataset path and format match wizard input
    soup_dict["data"]["train"] = data_path
    soup_dict["data"]["format"] = data_format

    # Pydantic validation
    try:
        validated = SoupConfig.model_validate(soup_dict)
        final_dict = validated.model_dump(exclude_unset=True, mode="json")
    except Exception:
        final_dict = soup_dict

    config_text = yaml.dump(final_dict, sort_keys=False, indent=2)

    # 5. Display Recipe Summary
    table = Table(title="Generated Fine-Tuning Recipe", border_style="cyan")
    table.add_column("Parameter", style="bold")
    table.add_column("Value", style="green")

    table.add_row("Base Model", base_model)
    table.add_row("Task", final_dict.get("task", target_task))
    table.add_row("Dataset", data_path)
    table.add_row("Format", data_format)
    table.add_row(
        "Hardware Target",
        f"{hw.device.upper()} ({'4-bit QLoRA' if hw.device == 'cuda' else 'Float16/Full'})",
    )
    table.add_row("Batch Size", str(final_dict.get("training", {}).get("batch_size", "auto")))
    table.add_row("Epochs", str(final_dict.get("training", {}).get("epochs", 3)))
    table.add_row("Output Dir", str(final_dict.get("output", "./output")))
    console.print(table)

    # 6. Save and finish
    output_path.write_text(config_text, encoding="utf-8")
    console.print(
        Panel(
            f"[bold green]✓ Config successfully saved to {output_path}[/]\n\n"
            f"Start training with:\n"
            f"  [bold cyan]soup train --config {output_path}[/]",
            title="Setup Complete",
            border_style="green",
        )
    )


def _interactive_wizard() -> str:
    """Walk user through config creation."""
    console.print(Panel("[bold]Soup Config Wizard[/]", subtitle="Let's set up your training"))

    base_model = Prompt.ask(
        "Base model",
        default="meta-llama/Llama-3.1-8B-Instruct",
    )
    task = Prompt.ask(
        "Task",
        choices=[
            "sft", "dpo", "kto", "orpo", "simpo", "ipo", "grpo", "ppo",
            "reward_model", "pretrain", "embedding",
        ],
        default="sft",
    )
    data_path = Prompt.ask("Training data path", default="./data/train.jsonl")

    # Preference tasks have fixed data formats — skip format prompt
    if task in ("dpo", "orpo", "simpo", "ipo"):
        data_format = "dpo"
    elif task == "kto":
        data_format = "kto"
    elif task == "pretrain":
        data_format = "plaintext"
    elif task == "embedding":
        data_format = "embedding"
    else:
        data_format = Prompt.ask(
            "Data format", choices=["alpaca", "sharegpt", "chatml"], default="alpaca",
        )
    epochs = Prompt.ask("Epochs", default="3")
    use_qlora = Prompt.ask("Use QLoRA (4-bit)?", choices=["yes", "no"], default="yes")

    quantization = "4bit" if use_qlora == "yes" else "none"

    task_block = ""
    if task == "grpo":
        reward_fn = Prompt.ask(
            "Reward function", choices=["accuracy", "format", "custom"], default="accuracy",
        )
        if reward_fn == "custom":
            reward_fn = Prompt.ask("Path to reward .py file", default="./reward.py")
        task_block = f"""  grpo_beta: 0.1
  num_generations: 4
  reward_fn: {reward_fn}
"""
    elif task == "kto":
        task_block = """  kto_beta: 0.1
"""
    elif task == "orpo":
        task_block = """  orpo_beta: 0.1
"""
    elif task == "simpo":
        task_block = """  simpo_gamma: 0.5
  cpo_alpha: 1.0
"""
    elif task == "ipo":
        task_block = """  ipo_tau: 0.1
"""
    elif task == "embedding":
        task_block = """  embedding_loss: contrastive
  embedding_margin: 0.5
  embedding_pooling: mean
"""
    elif task == "ppo":
        reward_model_path = Prompt.ask(
            "Reward model path", default="./output_rm",
        )
        task_block = f"""  reward_model: {reward_model_path}
  ppo_epochs: 4
  ppo_clip_ratio: 0.2
  ppo_kl_penalty: 0.05
"""

    return f"""# Soup training config
# Docs: https://github.com/MakazhanAlpamys/Soup

base: {base_model}
task: {task}

data:
  train: {data_path}
  format: {data_format}
  val_split: 0.1

training:
  epochs: {epochs}
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
    target_modules: auto
  quantization: {quantization}
{task_block}
output: ./output
"""
