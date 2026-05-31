"""Verifiable-reward synth data template (RLVR — Part C of v0.25.0)."""

SYSTEM_PROMPT = (
    "You are a training data generator for RLVR (RL from Verifiable Rewards). "
    "Generate problems with deterministic ground-truth answers that can be "
    "automatically verified (math, code, or JSON schema conformance)."
)

TEMPLATE_SPEC = {
    "domains": {
        "math": [
            "arithmetic",
            "algebra",
            "geometry",
            "word problem",
        ],
        "code": [
            "print statement",
            "string manipulation",
            "list operation",
            "arithmetic function",
        ],
        "json_schema": [
            "user profile",
            "product catalog entry",
            "event record",
            "API response object",
        ],
    },
}


def build_prompt(
    count: int,
    fmt: str,
    format_spec: str,
    domain: str = "math",
) -> str:
    """Build a generation prompt for a verifiable-reward domain.

    Args:
        count: Number of examples to generate.
        fmt: Output format (alpaca / sharegpt / chatml).
        format_spec: Format specification string embedded in prompt.
        domain: One of ``"math"``, ``"code"``, ``"json_schema"``.

    Returns:
        Complete generation prompt string.
    """
    sub_types = TEMPLATE_SPEC["domains"].get(
        domain,
        TEMPLATE_SPEC["domains"]["math"],
    )
    sub_list = "\n".join(f"  - {s}" for s in sub_types)

    if domain == "math":
        body = (
            "Each example is a math problem with a single numeric answer "
            "(integer or decimal). Put the answer after '####' on its own line."
        )
    elif domain == "code":
        body = (
            "Each example asks the model to write a short Python snippet that "
            "prints exactly one expected string or number. The expected stdout "
            "must be deterministic and reproducible."
        )
    else:  # json_schema
        body = (
            "Each example asks the model to produce a JSON object conforming "
            "to a given JSON Schema. Include the schema in the prompt and a "
            "valid example as the expected answer."
        )

    return (
        f"You are a training data generator. Generate exactly {count} diverse, "
        f"high-quality training examples for RLVR ({domain} domain).\n\n"
        f"Sub-types:\n{sub_list}\n\n"
        f"{body}\n\n"
        f"Format: {format_spec}\n\n"
        f"Return ONLY a JSON array of {count} examples. No markdown, no explanation."
    )
