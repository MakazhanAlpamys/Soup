"""Config values Soup loaded, with a warning, for one release before refusing them.

A value that never took effect is a defect to remove, but refusing it at once
breaks the configs Soup itself wrote while that value was the default:
``soup autopilot`` and every run-config dump write the whole model, defaults
included. So such a value first loaded with a warning naming the release that
would refuse it, the same staging #627 used for unknown keys (#759, #795, #1418).
That release has shipped: the values are now refused.

A schema validator calls :func:`warn_deprecated_value`. While
:data:`DEPRECATED_VALUE_SEVERITY` is ``"error"`` (as it is now) the call raises
``ValueError``, so the value is refused wherever it is validated. Under
``"warn"`` the loader turned each warning into one printed line per load, and
code that built ``SoupConfig`` directly saw an ordinary
:class:`SoupConfigDeprecationWarning`; that path stays, for the next value that
needs a release of notice.

The fields no code reads and no warning ever named (#1814) are a later cohort: they
warn through :func:`warn_unconsumed_field` whatever that switch says, and are refused
in :data:`UNCONSUMED_FIELD_REJECTION_VERSION`.
"""

from __future__ import annotations

import warnings

#: The release that stops warning about these values and refuses them.
#:
#: Written out once, here, and read by every warning message, because a
#: duplicated version is how a warning keeps promising a refusal after the
#: refusal shipped. The deadline test compares it against
#: ``soup_cli.__version__``, so the release that reaches it turns a test red
#: instead of turning the warning into a lie. One minor of notice: the warning
#: shipped in 0.76, the refusal in 0.77.
DEPRECATED_VALUE_REJECTION_VERSION = "0.77"

#: The release that refuses the declared fields no code reads and that nothing warned
#: about before v0.78.0: ``training.warmup_auto`` and ``training.forgetting_threshold``
#: (#1814). A cohort of its own, not :data:`DEPRECATED_VALUE_REJECTION_VERSION`: that
#: deadline has passed, and these fields get their own minor of notice (warned in 0.78,
#: refused in 0.79). Written out here, and built into every message by
#: :func:`warn_unconsumed_field`; ``tests/test_config_deprecated_fields.py`` pins it and
#: turns red in the release that reaches it.
UNCONSUMED_FIELD_REJECTION_VERSION = "0.79.0"

#: The release that removed ``soup export --format awq`` / ``gptq`` (#1710, #1951).
#:
#: A fact about the past, not a deadline, so it is not built from
#: :data:`DEPRECATED_VALUE_REJECTION_VERSION`: that constant moves with the next
#: deprecation, and the refusal would then date the removal to a release that
#: still shipped both formats. ``TestTheExportRemovalRelease`` in
#: ``tests/test_config_deprecation.py`` pins it and checks ``__version__`` has
#: reached it.
EXPORT_FORMATS_REMOVED_IN = "0.77.0"

#: What to do about a deprecated config value (#759, #795, #1418).
#:
#: ``"warn"``  -- load it, ignore it, and print one line naming the release
#:               that will refuse it.
#: ``"error"`` -- refuse it: the config load raises, and ``soup ab --beta``
#:               exits with a usage error.
#:
#: Mirrors ``STAGED_FIELD_SEVERITY`` in ``config/loader.py``. v0.76 shipped
#: ``"warn"``; the 0.77.0 release commit, which bumped ``__version__`` to
#: :data:`DEPRECATED_VALUE_REJECTION_VERSION`, flipped this to ``"error"``,
#: together with ``STAGED_FIELD_SEVERITY``. ``TestTheDeadline`` in
#: ``tests/test_config_deprecation.py`` pins both to the declared version. Read
#: it through :func:`refuses_deprecated_values`, at call time, never copied at
#: import.
DEPRECATED_VALUE_SEVERITY = "error"


class SoupConfigDeprecationWarning(FutureWarning):
    """A config value that loads, under ``"warn"``, and is refused in a named release."""


def refuses_deprecated_values() -> bool:
    """Whether a deprecated value is refused rather than warned about."""
    return DEPRECATED_VALUE_SEVERITY == "error"


def deadline_clause() -> str:
    """The sentence every deprecation message ends with, in either severity."""
    if refuses_deprecated_values():
        return f"Refused as of Soup v{DEPRECATED_VALUE_REJECTION_VERSION}."
    return f"Soup v{DEPRECATED_VALUE_REJECTION_VERSION} will refuse it."


def warn_unconsumed_field(setting: str, why: str, fix: str) -> None:
    """Warn that ``setting`` is accepted and read by nothing (#1814).

    ``setting`` is the line as written (``training.warmup_auto: true``), ``why`` says
    what reads it instead or why nothing does, and ``fix`` is the sentence that ends
    with what the user does. Always a warning, whatever
    :data:`DEPRECATED_VALUE_SEVERITY` says: that switch belongs to the v0.77 cohort and
    flipped with it, while this cohort is refused in
    :data:`UNCONSUMED_FIELD_REJECTION_VERSION`. The loader prints it once per load.
    """
    warnings.warn(
        f"{setting} has no effect: {why}. {fix} "
        f"Soup v{UNCONSUMED_FIELD_REJECTION_VERSION} will refuse it.",
        SoupConfigDeprecationWarning,
        stacklevel=3,
    )


def warn_deprecated_value(
    message: str, *, refusal: "str | None" = None, after_deadline: str = ""
) -> None:
    """Refuse the value ``message`` describes, or, under ``"warn"``, warn about it.

    Under ``"warn"`` this warns with ``message``. Under ``"error"`` it raises
    ``ValueError`` with ``refusal``, or with ``message`` when no refusal text is
    given: pass one whenever ``message`` says the value "is ignored", which is
    no longer true once it is refused. Either text should end with what the
    user does instead; the deadline clause is appended to both.

    ``after_deadline`` is the fix for another, older refusal of the same config
    that runs later: the refusal names it after the deadline clause, which
    belongs to this value alone, so that one edit loads. A warning does not
    carry it, since that later refusal still runs and names its own fix.
    """
    if refuses_deprecated_values():
        text = message if refusal is None else refusal
        tail = f" {after_deadline}" if after_deadline else ""
        raise ValueError(f"{text} {deadline_clause()}{tail}")
    warnings.warn(
        f"{message} {deadline_clause()}", SoupConfigDeprecationWarning, stacklevel=3
    )
