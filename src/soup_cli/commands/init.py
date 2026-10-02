"""soup init — interactive project setup wizard."""

import errno
import os
import stat
import tempfile
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt

from soup_cli.templates import list_templates, load_template
from soup_cli.utils.paths import open_no_follow, refuse_linked_dirs
from soup_cli.utils.terminal import for_terminal

console = Console()


def _print_link_refusal(output: str) -> None:
    console.print(
        f"[red]{for_terminal(output)} is a symbolic link, junction or other reparse "
        "point; soup init does not write through it.[/]\n"
        "Remove it or choose a different --output."
    )


def _refuse_link_at(output: str) -> None:
    """Exit 1 when the starter config path is a symbolic link, junction or other reparse point.

    Only the target itself is inspected (``stop_at`` is its own parent); the
    directories above it are where the user chose to write.
    """
    output_path = Path(output)
    try:
        refuse_linked_dirs(output_path, stop_at=output_path.parent)
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            raise
        _print_link_refusal(output)
        raise typer.Exit(1) from exc


def _create_by_rename(output: str, config_text: str) -> None:
    """Create ``output`` on Windows without following anything that appears there.

    On Windows an exclusive create (``O_CREAT | O_EXCL``) still follows a
    dangling symbolic link and creates the link's target. A rename does not:
    ``os.rename`` fails on Windows when anything, a link or junction included,
    exists at the destination. So the config is written to a staging file next
    to ``output`` and renamed into place; mode and bytes match a direct write.
    """
    parent = os.path.dirname(os.path.abspath(output))
    fd, staging = tempfile.mkstemp(prefix=".soup-init.", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(config_text)
        os.rename(staging, output)
    except BaseException:
        try:
            os.unlink(staging)
        except OSError:
            pass
        raise


def _write_config(output: str, config_text: str) -> None:
    """Write ``config_text`` to ``output`` without following a link there.

    Nothing is created or written through a link that appears at ``output``
    after :func:`_refuse_link_at` checked it:

    * a new file is created exclusively: ``O_CREAT | O_EXCL`` with
      ``O_NOFOLLOW`` through ``open_no_follow`` on POSIX, and
      :func:`_create_by_rename` on Windows;
    * an existing file is opened without ``O_CREAT`` through ``open_no_follow``
      (``O_NOFOLLOW`` on POSIX, an lstat/fstat cross-check on Windows), so a
      dangling link swapped in for it cannot get its target created, and it is
      truncated only after that check has passed, so a refused open leaves a
      link's target untouched.

    A failure is reported as the link refusal when the target is a link by
    then, and as itself otherwise. Mode, truncation and newline handling match
    the ``Path.write_text`` this replaces.
    """
    creating = not os.path.lexists(output)
    try:
        if creating and os.name == "nt":
            _create_by_rename(output, config_text)
            return
        flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
        if creating:
            flags |= os.O_CREAT | os.O_EXCL
        fd = open_no_follow(output, flags, 0o666)
    except OSError as exc:
        if exc.errno != errno.ELOOP:
            # The target may have changed after the check: refuse it if it is
            # a link now, otherwise report the original error.
            _refuse_link_at(output)
            raise
        _print_link_refusal(output)
        raise typer.Exit(1) from exc
    try:
        # O_TRUNC semantics: only a regular file is truncated.
        if stat.S_ISREG(os.fstat(fd).st_mode):
            os.ftruncate(fd, 0)
    except OSError:
        os.close(fd)
        raise
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        handle.write(config_text)


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
):
    """Create a new soup.yaml config interactively or from a template."""
    output_path = Path(output)
    _refuse_link_at(output)

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

    _write_config(output, config_text)
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
