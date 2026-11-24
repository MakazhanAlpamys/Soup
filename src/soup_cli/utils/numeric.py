"""Numeric conversion shared by configuration validators."""


def float_with_overflow_check(value: int | float, name: str) -> float:
    """Name an overflowing conversion without echoing its potentially huge input.

    Callers retain their existing type, finiteness and range checks. Only an
    overflow changes exception type so Pydantic can report a validation error.
    """
    try:
        return float(value)
    except OverflowError:
        raise ValueError(f"{name} must be finite") from None
