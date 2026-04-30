"""JinjaTemplateAnalyzer — discover which message fields a chat template uses.

Mirrors Axolotl's ``JinjaTemplateAnalyzer`` (used by their per-message
training-mask logic). Walks the parsed Jinja AST instead of regex so we
correctly handle both attribute-style ``{{ m.role }}`` and subscript-style
``{{ m["role"] }}`` access.

Used by:

* ``loss_mask`` (v0.36.0) — when ``train_on_messages_with_train_field`` is
  enabled, the analyzer confirms the chat template actually references
  ``message.train``; otherwise the flag would silently no-op.
* Future v0.37.0 multipack work — when assembling per-message labels, the
  analyzer tells the formatter which non-standard fields it must preserve
  through the pack-then-render pipeline.

Security:

* Templates are parsed via ``Environment.parse`` only — never rendered, so
  the analyzer cannot trigger SSRF / filesystem reads.
* Length-capped at 128KB to prevent DoS on a hand-crafted megabyte template.
* Null-byte rejection (matches v0.36.0 chat-template policy).
"""

from __future__ import annotations

from collections.abc import Set as AbstractSet

# Message fields the HF / OpenAI chat-completions schema treats as standard.
# Anything outside this set is "non-standard" and may need extra handling
# in the per-message training-label pipeline.
DEFAULT_MESSAGE_FIELDS: frozenset[str] = frozenset({"role", "content"})


_MAX_TEMPLATE_BYTES: int = 128 * 1024


def _validate_template_input(template: str) -> None:
    if not isinstance(template, str):
        raise TypeError(
            f"template must be str, got {type(template).__name__}"
        )
    if not template:
        raise ValueError("template must be non-empty")
    if "\x00" in template:
        raise ValueError("template must not contain null bytes")
    if len(template) > _MAX_TEMPLATE_BYTES:
        raise ValueError(
            f"template too large: {len(template)} bytes "
            f"(max {_MAX_TEMPLATE_BYTES})"
        )


def extract_message_fields(template: str) -> set[str]:
    """Walk ``template``'s Jinja AST and return the set of message fields.

    Detects both attribute-style ``{{ m.role }}`` and subscript-style
    ``{{ m["role"] }}`` access on any variable iterated from
    ``messages`` (e.g. ``{% for m in messages %}``).

    Args:
        template: Jinja2 source (chat-template).

    Returns:
        Set of field names referenced on per-message variables. Empty set
        if the template never iterates ``messages``.

    Raises:
        ValueError: empty, null-byte, oversize, or unparseable template.
        TypeError: non-string input.
    """
    _validate_template_input(template)

    from jinja2 import Environment, nodes
    from jinja2.exceptions import TemplateSyntaxError

    env = Environment(autoescape=False)  # noqa: S701 — analysis-only, never rendered
    try:
        ast = env.parse(template)
    except TemplateSyntaxError as exc:
        raise ValueError(f"failed to parse template: {exc}") from exc

    # Find every ``for X in messages`` loop, then scan the loop body for
    # X.<attr> and X["<attr>"] access.
    loop_var_names: set[str] = set()
    for for_node in ast.find_all(nodes.For):
        # ``iter`` is a Name("messages") in the typical case.
        iter_node = for_node.iter
        if isinstance(iter_node, nodes.Name) and iter_node.name == "messages":
            target = for_node.target
            if isinstance(target, nodes.Name):
                loop_var_names.add(target.name)

    if not loop_var_names:
        return set()

    fields: set[str] = set()
    # Walk the whole AST and collect attribute / subscript access on any
    # of our message loop variables.
    for getattr_node in ast.find_all(nodes.Getattr):
        if (
            isinstance(getattr_node.node, nodes.Name)
            and getattr_node.node.name in loop_var_names
        ):
            fields.add(getattr_node.attr)
    for getitem_node in ast.find_all(nodes.Getitem):
        if (
            isinstance(getitem_node.node, nodes.Name)
            and getitem_node.node.name in loop_var_names
            and isinstance(getitem_node.arg, nodes.Const)
            and isinstance(getitem_node.arg.value, str)
        ):
            fields.add(getitem_node.arg.value)
    return fields


class JinjaTemplateAnalyzer:
    """Cached message-field analysis for a chat template.

    Construct once per template; query repeatedly via :meth:`has_field`.
    """

    def __init__(self, template: str) -> None:
        self._template = template
        self._fields: set[str] = extract_message_fields(template)

    @property
    def message_fields(self) -> set[str]:
        """Snapshot copy of the discovered message fields."""
        return set(self._fields)

    def has_field(self, name: str) -> bool:
        """Return True if the template references ``message.<name>``."""
        if not isinstance(name, str):
            raise TypeError(
                f"name must be str, got {type(name).__name__}"
            )
        return name in self._fields

    def non_standard_fields(
        self, standard: AbstractSet[str] = DEFAULT_MESSAGE_FIELDS,
    ) -> set[str]:
        """Return fields used by the template that are NOT in ``standard``."""
        return self._fields - set(standard)
