"""soup init — interactive project setup wizard."""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soup_cli.templates import list_templates, load_template

console = Console()


def init(
    template: str = typer.Option(
        None,
        "--template",
        "-t",
        help="Template: chat, code, medical, reasoning, vision, audio, rlhf, "
        "kto, orpo, simpo, ipo, pretrain, moe, embedding, longcontext",
    ),
    output: str = typer.Option(
        "soup.yaml",
        "--output",
        "-o",
        help="Output config file path",
    ),
):
    """Create a new soup.yaml config interactively or from a template."""
    output_path = Path(output)

    if output_path.exists():
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
