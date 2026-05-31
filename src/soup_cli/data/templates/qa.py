"""Question-answer from context template for synthetic data generation."""

SYSTEM_PROMPT = (
    "You are a QA training data generator. Generate question-answer pairs "
    "that are grounded in provided context documents."
)


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    context: str = "",
) -> str:
    """Build the generation prompt for QA pairs.

    Args:
        count: Number of examples to generate.
        fmt: Output format (alpaca/sharegpt/chatml).
        format_spec: Format specification string.
        context: Source document text to generate QA from.

    Returns:
        Complete generation prompt string.
    """
    context_section = ""
    if context:
        # Cap context to prevent prompt overflow
        truncated = context[:8000]
        context_section = (
            f"\n\nSource document to generate questions from:\n"
            f"---\n{truncated}\n---\n\n"
            f"Generate questions and answers that are grounded in this document. "
            f"Answers must be derivable from the text."
        )
    else:
        context_section = (
            "\n\nGenerate diverse questions and detailed answers on "
            "general knowledge topics."
        )

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality question-answer pairs.{context_section}\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
