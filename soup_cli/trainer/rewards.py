"""Reward functions for GRPO training.

Built-in reward functions:
  - accuracy: checks if the model answer matches the expected answer
  - format: checks if the response follows a structured format (e.g., <think>...</think>)

Custom reward functions can be loaded from a Python file with a
`reward_fn(completions, **kwargs)` callable.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

from rich.console import Console

console = Console()


def accuracy_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward based on whether the final answer matches the expected answer.

    Looks for the answer after the last '####' or in a \\boxed{} block.
    Falls back to checking if the expected answer appears anywhere in the response.

    Args:
        completions: list of message lists, each containing a completion with 'content'.
        **kwargs: must contain 'answer' — the expected answer for each prompt.

    Returns:
        List of float rewards (1.0 for correct, 0.0 for incorrect).
    """
    answers = kwargs.get("answer", [])
    rewards = []
    for completion, expected in zip(completions, answers):
        content = completion[-1]["content"] if completion else ""
        predicted = _extract_answer(content)
        if predicted is not None and predicted.strip() == str(expected).strip():
            rewards.append(1.0)
        elif str(expected).strip().lower() in content.lower():
            rewards.append(0.5)
        else:
            rewards.append(0.0)
    return rewards


def format_reward(completions: list[list[dict]], **kwargs) -> list[float]:
    """Reward based on whether the response follows a structured reasoning format.

    Checks for:
      - <think>...</think> block (chain-of-thought)
      - A final answer section after the thinking block

    Args:
        completions: list of message lists.
        **kwargs: unused.

    Returns:
        List of float rewards (0.0 to 1.0).
    """
    rewards = []
    for completion in completions:
        content = completion[-1]["content"] if completion else ""
        score = 0.0
        # Check for <think> block
        if re.search(r"<think>.*?</think>", content, re.DOTALL):
            score += 0.5
        # Check for content after </think>
        after_think = re.split(r"</think>", content)
        if len(after_think) > 1 and after_think[-1].strip():
            score += 0.5
        rewards.append(score)
    return rewards


def _extract_answer(text: str) -> str | None:
    """Extract the final answer from model output.

    Supports:
      - #### <answer> format (GSM8K style)
      - \\boxed{<answer>} format (math style)
    """
    # Try #### format
    parts = text.split("####")
    if len(parts) > 1:
        return parts[-1].strip()
    # Try \\boxed{} format
    match = re.search(r"\\boxed\{([^}]+)\}", text)
    if match:
        return match.group(1).strip()
    return None


# Registry of built-in reward functions
BUILTIN_REWARDS: dict[str, callable] = {
    "accuracy": accuracy_reward,
    "format": format_reward,
}


def load_reward_fn(reward_fn_spec: str) -> callable:
    """Load a reward function by name or from a custom Python file.

    Args:
        reward_fn_spec: Either a built-in name ('accuracy', 'format') or
                        a path to a .py file containing a `reward_fn` callable.

    Returns:
        A callable reward function with signature:
        (completions: list[list[dict]], **kwargs) -> list[float]
    """
    # Built-in reward function
    if reward_fn_spec in BUILTIN_REWARDS:
        console.print(f"[dim]Using built-in reward function: {reward_fn_spec}[/]")
        return BUILTIN_REWARDS[reward_fn_spec]

    # Custom Python file
    reward_path = Path(reward_fn_spec)
    if reward_path.exists() and reward_path.suffix == ".py":
        console.print(
            f"[bold yellow]Warning:[/] Loading custom reward function from: "
            f"[bold]{reward_path.resolve()}[/]\n"
            f"[yellow]This will execute arbitrary Python code. "
            f"Only use reward files you trust.[/]"
        )
        spec = importlib.util.spec_from_file_location("custom_reward", reward_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if not hasattr(module, "reward_fn"):
            raise ValueError(
                f"Custom reward file {reward_path} must define a 'reward_fn' callable.\n"
                f"Example:\n"
                f"  def reward_fn(completions, **kwargs):\n"
                f"      return [1.0] * len(completions)"
            )
        return module.reward_fn

    raise ValueError(
        f"Unknown reward function: '{reward_fn_spec}'\n"
        f"Options: {', '.join(BUILTIN_REWARDS.keys())} or path to a .py file"
    )
