"""Code instruction pair template for synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a coding instruction data generator. Generate diverse, high-quality "
    "code instruction-response pairs covering different programming tasks."
)

TEMPLATE_SPEC = {
    "languages": ["Python", "JavaScript", "Go", "Rust", "Java"],
    "types": ["function", "debug", "explain", "refactor", "test"],
}


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    language: str = "Python",
    task_type: str = "function",
) -> str:
    """Build the generation prompt for code instruction pairs.

    Args:
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        format_spec: Format specification string.
        language: Programming language to focus on.
        task_type: Type of coding task.

    Returns:
        Complete generation prompt string.
    """
    type_descriptions = {
        "function": "writing functions to solve specific problems",
        "debug": "finding and fixing bugs in code snippets",
        "explain": "explaining what given code does step by step",
        "refactor": "improving and refactoring existing code",
        "test": "writing unit tests for given functions",
    }

    task_desc = type_descriptions.get(task_type, type_descriptions["function"])

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality coding instruction-response pairs.\n\n"
        f"Language: {language}\n"
        f"Task type: {task_desc}\n\n"
        f"Each example should involve {language} code for {task_desc}.\n"
        f"Include realistic code snippets in responses.\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
