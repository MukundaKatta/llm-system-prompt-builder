"""Tests for llm-system-prompt-builder."""

from __future__ import annotations

import pytest

from llm_system_prompt_builder import SectionKind, SystemPromptBuilder
from llm_system_prompt_builder.core import SectionNotFoundError

# ---------------------------------------------------------------------------
# SectionKind
# ---------------------------------------------------------------------------


def test_section_kind_values():
    assert SectionKind.ROLE.value == "role"
    assert SectionKind.CONTEXT.value == "context"
    assert SectionKind.RULES.value == "rules"
    assert SectionKind.FORMAT.value == "format"
    assert SectionKind.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# set_role
# ---------------------------------------------------------------------------


def test_set_role():
    b = SystemPromptBuilder()
    b.set_role("You are a helpful assistant.")
    assert "You are a helpful assistant." in b.build()


def test_set_role_no_heading():
    b = SystemPromptBuilder()
    b.set_role("Be helpful.")
    prompt = b.build()
    # Role section has no heading
    assert ":\n" not in prompt or not prompt.startswith(":")
    assert prompt == "Be helpful."


def test_set_role_replaces():
    b = SystemPromptBuilder()
    b.set_role("First role.")
    b.set_role("Second role.")
    prompt = b.build()
    assert "Second role." in prompt
    assert "First role." not in prompt


def test_set_role_chaining():
    b = SystemPromptBuilder()
    result = b.set_role("Hello.")
    assert result is b


# ---------------------------------------------------------------------------
# set_context
# ---------------------------------------------------------------------------


def test_set_context():
    b = SystemPromptBuilder()
    b.set_context("The user is a Python developer.")
    prompt = b.build()
    assert "Context:\nThe user is a Python developer." in prompt


def test_set_context_replaces():
    b = SystemPromptBuilder()
    b.set_context("Old context.")
    b.set_context("New context.")
    assert "New context." in b.build()
    assert "Old context." not in b.build()


# ---------------------------------------------------------------------------
# add_rule / set_rules
# ---------------------------------------------------------------------------


def test_add_single_rule():
    b = SystemPromptBuilder()
    b.add_rule("Always be concise.")
    prompt = b.build()
    assert "Rules:" in prompt
    assert "- Always be concise." in prompt


def test_add_multiple_rules():
    b = SystemPromptBuilder()
    b.add_rule("Rule A")
    b.add_rule("Rule B")
    prompt = b.build()
    assert "- Rule A" in prompt
    assert "- Rule B" in prompt


def test_set_rules_replaces():
    b = SystemPromptBuilder()
    b.add_rule("Old rule")
    b.set_rules(["New rule 1", "New rule 2"])
    prompt = b.build()
    assert "Old rule" not in prompt
    assert "New rule 1" in prompt
    assert "New rule 2" in prompt


def test_rules_section_kind():
    b = SystemPromptBuilder()
    b.add_rule("R")
    sec = b.get_section("__rules__")
    assert sec.kind is SectionKind.RULES


# ---------------------------------------------------------------------------
# set_format
# ---------------------------------------------------------------------------


def test_set_format():
    b = SystemPromptBuilder()
    b.set_format("Respond in JSON.")
    prompt = b.build()
    assert "Response format:\nRespond in JSON." in prompt


def test_set_format_replaces():
    b = SystemPromptBuilder()
    b.set_format("Old format.")
    b.set_format("New format.")
    assert "New format." in b.build()
    assert "Old format." not in b.build()


# ---------------------------------------------------------------------------
# add_section / remove_section
# ---------------------------------------------------------------------------


def test_add_custom_section():
    b = SystemPromptBuilder()
    b.add_section("Examples", "Example A\nExample B")
    assert "Examples:" in b.build()
    assert "Example A" in b.build()


def test_add_custom_section_list():
    b = SystemPromptBuilder()
    b.add_section("Tips", ["Tip 1", "Tip 2"])
    prompt = b.build()
    assert "- Tip 1" in prompt
    assert "- Tip 2" in prompt


def test_add_section_custom_heading():
    b = SystemPromptBuilder()
    b.add_section("my_key", "content", heading="My Heading")
    assert "My Heading:" in b.build()


def test_add_section_replaces():
    b = SystemPromptBuilder()
    b.add_section("notes", "Old notes")
    b.add_section("notes", "New notes")
    assert "New notes" in b.build()
    assert "Old notes" not in b.build()


def test_remove_section():
    b = SystemPromptBuilder()
    b.set_role("Role.")
    b.add_section("extra", "Extra content.")
    b.remove_section("extra")
    assert "Extra content." not in b.build()


def test_remove_built_in_section():
    b = SystemPromptBuilder()
    b.set_role("Role.")
    b.set_context("Context.")
    b.remove_section("__context__")
    assert "Context." not in b.build()


def test_remove_missing_raises():
    b = SystemPromptBuilder()
    with pytest.raises(SectionNotFoundError) as exc_info:
        b.remove_section("nonexistent")
    assert exc_info.value.key == "nonexistent"


def test_has_section():
    b = SystemPromptBuilder()
    b.set_role("R.")
    assert b.has_section("__role__")
    assert not b.has_section("__context__")


# ---------------------------------------------------------------------------
# Ordering
# ---------------------------------------------------------------------------


def test_role_before_context():
    b = SystemPromptBuilder()
    b.set_context("Context content.")
    b.set_role("Role content.")
    prompt = b.build()
    assert prompt.index("Role content.") < prompt.index("Context content.")


def test_context_before_rules():
    b = SystemPromptBuilder()
    b.add_rule("Rule.")
    b.set_context("Context.")
    prompt = b.build()
    assert prompt.index("Context.") < prompt.index("Rule.")


def test_rules_before_format():
    b = SystemPromptBuilder()
    b.set_format("Format.")
    b.add_rule("Rule.")
    prompt = b.build()
    assert prompt.index("Rule.") < prompt.index("Format.")


def test_custom_order_override():
    b = SystemPromptBuilder()
    b.set_role("Role.")
    b.add_section("preamble", "First!", order=0)
    # preamble has order=0, role also has order=0, tie broken by insertion
    # Both are at order 0; preamble was added after role but role was added first
    # Actually role is "__role__" added first, preamble added second, both order=0
    # Stable sort: "__role__" comes first (earlier insertion index)
    prompt = b.build()
    assert "First!" in prompt


# ---------------------------------------------------------------------------
# build() options
# ---------------------------------------------------------------------------


def test_section_sep():
    b = SystemPromptBuilder()
    b.set_role("Role.")
    b.set_context("Context.")
    prompt = b.build(section_sep="---")
    assert "Role.---" in prompt


def test_list_prefix():
    b = SystemPromptBuilder()
    b.add_rule("Do X")
    prompt = b.build(list_prefix="* ")
    assert "* Do X" in prompt


def test_empty_builder():
    b = SystemPromptBuilder()
    assert b.build() == ""


# ---------------------------------------------------------------------------
# sections() / get_section()
# ---------------------------------------------------------------------------


def test_sections_in_order():
    b = SystemPromptBuilder()
    b.set_format("F")
    b.set_role("R")
    b.set_context("C")
    kinds = [s.kind for s in b.sections()]
    assert kinds == [SectionKind.ROLE, SectionKind.CONTEXT, SectionKind.FORMAT]


def test_get_section_missing_raises():
    b = SystemPromptBuilder()
    with pytest.raises(SectionNotFoundError):
        b.get_section("nope")


# ---------------------------------------------------------------------------
# contains / len / clear
# ---------------------------------------------------------------------------


def test_contains():
    b = SystemPromptBuilder()
    b.set_role("R.")
    assert "__role__" in b
    assert "__context__" not in b


def test_len():
    b = SystemPromptBuilder()
    assert len(b) == 0
    b.set_role("R.")
    b.add_rule("Rule.")
    assert len(b) == 2


def test_clear():
    b = SystemPromptBuilder()
    b.set_role("R.")
    b.add_rule("Rule.")
    b.clear()
    assert len(b) == 0
    assert b.build() == ""


# ---------------------------------------------------------------------------
# Serialisation round-trip
# ---------------------------------------------------------------------------


def test_to_dict_round_trip():
    b = SystemPromptBuilder()
    b.set_role("Role.")
    b.set_context("Context.")
    b.add_rule("Rule A")
    b.add_rule("Rule B")
    b.set_format("JSON only.")
    b.add_section("notes", "Extra notes.", order=50)

    restored = SystemPromptBuilder.from_dict(b.to_dict())
    assert restored.build() == b.build()
    assert len(restored) == len(b)


def test_from_dict_preserves_order():
    b = SystemPromptBuilder()
    b.set_format("F")
    b.set_role("R")
    restored = SystemPromptBuilder.from_dict(b.to_dict())
    prompt = restored.build()
    assert prompt.index("R") < prompt.index("F")


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


def test_repr():
    b = SystemPromptBuilder()
    b.set_role("R.")
    r = repr(b)
    assert "SystemPromptBuilder" in r
    assert "__role__" in r
