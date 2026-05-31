"""Reasoning / chain-of-thought template for GRPO synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a reasoning training data generator. Generate problems with "
    "detailed step-by-step solutions for training reasoning models."
)

DOMAINS = ("math", "logic", "code")


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    domain: str = "math",
) -> str:
    """Build the generation prompt for reasoning data.

    Args:
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        format_spec: Format specification string.
        domain: Problem domain (math/logic/code).

    Returns:
        Complete generation prompt string.
    """
    domain_descriptions = {
        "math": (
            "mathematical problems requiring multi-step calculation. "
            "Include algebra, arithmetic, geometry, and word problems."
        ),
        "logic": (
            "logical reasoning puzzles and deduction problems. "
            "Include syllogisms, truth tables, and constraint satisfaction."
        ),
        "code": (
            "programming challenges requiring algorithmic thinking. "
            "Include data structures, algorithms, and problem decomposition."
        ),
    }

    domain_desc = domain_descriptions.get(domain, domain_descriptions["math"])

    return (
        f"You are a training data generator. Generate exactly {count} diverse "
        f"reasoning problems with step-by-step solutions.\n\n"
        f"Domain: {domain_desc}\n\n"
        f"Each solution must show detailed chain-of-thought reasoning. "
        f"Use <think>...</think> tags to wrap the reasoning steps, then provide "
        f"the final answer.\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
