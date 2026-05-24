"""Structured multi-section system prompt builder for LLMs.

:class:`SystemPromptBuilder` composes a system prompt from named sections
rendered in a predictable order.  Built-in sections cover the most common
prompt engineering patterns — role, context, rules, and format — while
:meth:`~SystemPromptBuilder.add_section` supports arbitrary custom sections.

Example::

    builder = SystemPromptBuilder()
    builder.set_role("You are a concise coding assistant.")
    builder.add_rule("Always explain your reasoning.")
    builder.add_rule("Prefer Python 3.10+ idioms.")
    builder.set_format("Respond with a code block, then a one-sentence summary.")

    prompt = builder.build()
    # "You are a concise coding assistant.\\n\\nRules:\\n- Always...\\n\\n..."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SectionKind(str, Enum):
    """Kind of a prompt section."""

    ROLE = "role"
    CONTEXT = "context"
    RULES = "rules"
    FORMAT = "format"
    CUSTOM = "custom"


# Default render order for built-in section kinds.
_KIND_ORDER: dict[SectionKind, int] = {
    SectionKind.ROLE: 0,
    SectionKind.CONTEXT: 10,
    SectionKind.RULES: 20,
    SectionKind.FORMAT: 30,
    SectionKind.CUSTOM: 40,
}


class SectionNotFoundError(KeyError):
    """Raised when a section key is not found."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"Section {key!r} not found.")


@dataclass
class _Section:
    """Internal representation of a single prompt section.

    Attributes:
        key:     Unique lookup key (not rendered).
        heading: Displayed heading (empty string = no heading).
        kind:    Section kind.
        content: String or list of strings (list → bullet list).
        order:   Render position (lower = earlier).
    """

    key: str
    heading: str
    kind: SectionKind
    content: str | list[str]
    order: int = field(default=40)

    def render(self, *, list_prefix: str = "- ") -> str:
        """Render this section to a string fragment."""
        if isinstance(self.content, list):
            body = "\n".join(f"{list_prefix}{item}" for item in self.content)
        else:
            body = self.content
        if self.heading:
            return f"{self.heading}:\n{body}"
        return body

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "heading": self.heading,
            "kind": self.kind.value,
            "content": self.content,
            "order": self.order,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> _Section:
        return cls(
            key=data["key"],
            heading=data.get("heading", ""),
            kind=SectionKind(data.get("kind", SectionKind.CUSTOM.value)),
            content=data["content"],
            order=int(data.get("order", 40)),
        )


class SystemPromptBuilder:
    """Compose a system prompt from ordered named sections.

    Built-in section helpers: :meth:`set_role`, :meth:`set_context`,
    :meth:`add_rule` / :meth:`set_rules`, :meth:`set_format`.
    Use :meth:`add_section` for fully custom sections.

    Sections are emitted in ascending *order* value.  Built-in sections
    default to: role (0) < context (10) < rules (20) < format (30).
    Custom sections default to order 40, or you can pass an explicit value.

    Example::

        builder = SystemPromptBuilder()
        builder.set_role("You are a travel agent.")
        builder.add_rule("Always confirm destination before booking.")
        prompt = builder.build()
    """

    def __init__(self) -> None:
        # key → _Section (insertion order preserved for stable sort tiebreak)
        self._sections: dict[str, _Section] = {}

    # ------------------------------------------------------------------
    # Built-in section helpers
    # ------------------------------------------------------------------

    def set_role(self, text: str) -> SystemPromptBuilder:
        """Set the role section (no heading; appears first).

        Replaces any previously set role.
        """
        self._put(
            key="__role__",
            heading="",
            kind=SectionKind.ROLE,
            content=text,
            order=_KIND_ORDER[SectionKind.ROLE],
        )
        return self

    def set_context(self, text: str) -> SystemPromptBuilder:
        """Set the context section (heading: "Context").

        Replaces any previously set context.
        """
        self._put(
            key="__context__",
            heading="Context",
            kind=SectionKind.CONTEXT,
            content=text,
            order=_KIND_ORDER[SectionKind.CONTEXT],
        )
        return self

    def add_rule(self, rule: str) -> SystemPromptBuilder:
        """Append one rule to the rules list."""
        key = "__rules__"
        if key in self._sections:
            existing = self._sections[key].content
            if isinstance(existing, list):
                existing.append(rule)
            else:
                self._sections[key].content = [existing, rule]
        else:
            self._put(
                key=key,
                heading="Rules",
                kind=SectionKind.RULES,
                content=[rule],
                order=_KIND_ORDER[SectionKind.RULES],
            )
        return self

    def set_rules(self, rules: list[str]) -> SystemPromptBuilder:
        """Replace the entire rules list."""
        self._put(
            key="__rules__",
            heading="Rules",
            kind=SectionKind.RULES,
            content=list(rules),
            order=_KIND_ORDER[SectionKind.RULES],
        )
        return self

    def set_format(self, text: str) -> SystemPromptBuilder:
        """Set the response-format section (heading: "Response format").

        Replaces any previously set format.
        """
        self._put(
            key="__format__",
            heading="Response format",
            kind=SectionKind.FORMAT,
            content=text,
            order=_KIND_ORDER[SectionKind.FORMAT],
        )
        return self

    # ------------------------------------------------------------------
    # Custom sections
    # ------------------------------------------------------------------

    def add_section(
        self,
        key: str,
        content: str | list[str],
        *,
        order: int | None = None,
        heading: str | None = None,
    ) -> SystemPromptBuilder:
        """Add or replace an arbitrary section.

        Args:
            key:     Unique lookup key (also used as default heading).
            content: String or list of strings (list → bullet list).
            order:   Render position; defaults to 40.
            heading: Override heading text.  Defaults to *key*.

        Returns:
            ``self`` for chaining.
        """
        self._put(
            key=key,
            heading=heading if heading is not None else key,
            kind=SectionKind.CUSTOM,
            content=content,
            order=order if order is not None else _KIND_ORDER[SectionKind.CUSTOM],
        )
        return self

    def remove_section(self, key: str) -> None:
        """Remove a section by *key*.

        Raises:
            SectionNotFoundError: If *key* is not present.
        """
        if key not in self._sections:
            raise SectionNotFoundError(key)
        del self._sections[key]

    def has_section(self, key: str) -> bool:
        """Return ``True`` if a section with *key* exists."""
        return key in self._sections

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(
        self,
        *,
        section_sep: str = "\n\n",
        list_prefix: str = "- ",
    ) -> str:
        """Render all sections to a single system prompt string.

        Sections are emitted in ascending order value, with ties broken by
        insertion order (stable sort).

        Args:
            section_sep: String inserted between sections.
            list_prefix: Prefix for list items (default ``"- "``).

        Returns:
            The complete system prompt string.
        """
        ordered = sorted(
            enumerate(self._sections.values()),
            key=lambda x: (x[1].order, x[0]),
        )
        parts = [
            sec.render(list_prefix=list_prefix)
            for _, sec in ordered
            if sec.content or sec.content == ""
        ]
        return section_sep.join(p for p in parts if p)

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def sections(self) -> list[_Section]:
        """All sections in render order (ascending order value, FIFO tiebreak)."""
        return [
            sec
            for _, sec in sorted(
                enumerate(self._sections.values()),
                key=lambda x: (x[1].order, x[0]),
            )
        ]

    def get_section(self, key: str) -> _Section:
        """Return section by *key*.

        Raises:
            SectionNotFoundError: If not found.
        """
        if key not in self._sections:
            raise SectionNotFoundError(key)
        return self._sections[key]

    def clear(self) -> None:
        """Remove all sections."""
        self._sections.clear()

    def __len__(self) -> int:
        return len(self._sections)

    def __contains__(self, key: str) -> bool:
        return key in self._sections

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict (preserves insertion order)."""
        return {
            "sections": [sec.to_dict() for sec in self._sections.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SystemPromptBuilder:
        """Reconstruct a :class:`SystemPromptBuilder` from a plain dict."""
        builder = cls()
        for d in data.get("sections", []):
            sec = _Section.from_dict(d)
            builder._sections[sec.key] = sec
        return builder

    def __repr__(self) -> str:
        keys = list(self._sections.keys())
        return f"SystemPromptBuilder(sections={keys!r})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _put(
        self,
        *,
        key: str,
        heading: str,
        kind: SectionKind,
        content: str | list[str],
        order: int,
    ) -> None:
        self._sections[key] = _Section(
            key=key,
            heading=heading,
            kind=kind,
            content=content,
            order=order,
        )
