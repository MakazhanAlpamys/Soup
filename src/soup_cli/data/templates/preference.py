"""Preference data template for DPO/KTO/ORPO synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a preference training data generator. Generate examples with both "
    "good (chosen) and bad (rejected) responses for preference learning."
)


def build_prompt(
    count: int,
    task: str = "dpo",
) -> str:
    """Build the generation prompt for preference data.

    Args:
        count: Number of examples to generate.
        task: Target task format (dpo/kto/orpo).

    Returns:
        Complete generation prompt string.
    """
    if task == "kto":
        format_desc = (
            'Each example must be a JSON object with keys: '
            '"prompt" (the user question), '
            '"completion" (the model response), '
            '"label" (boolean: true for good, false for bad). '
            'Generate roughly equal numbers of true and false labels.'
        )
    else:
        # DPO / ORPO format
        format_desc = (
            'Each example must be a JSON object with keys: '
            '"prompt" (the user question), '
            '"chosen" (a good, helpful response), '
            '"rejected" (a bad, unhelpful, or incorrect response). '
            'The chosen response should be clearly better than the rejected one.'
        )

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality preference training examples.\n\n"
        f"Format: {format_desc}\n\n"
        f"Cover diverse topics: math, science, coding, writing, reasoning.\n"
        f"Make rejected responses subtly wrong (not obviously garbage).\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
