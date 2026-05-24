# llm-system-prompt-builder

Structured multi-section system prompt builder for LLMs.

Compose prompts from named sections (role, context, rules, format, custom) rendered in a predictable order.

## Install

```bash
pip install llm-system-prompt-builder
```

## Quick start

```python
from llm_system_prompt_builder import SystemPromptBuilder

builder = SystemPromptBuilder()
builder.set_role("You are a concise coding assistant.")
builder.set_context("The user is working on a FastAPI project.")
builder.add_rule("Always explain your reasoning.")
builder.add_rule("Prefer Python 3.10+ idioms.")
builder.set_format("Respond with a code block, then a one-sentence summary.")

prompt = builder.build()
```

Output:
```
You are a concise coding assistant.

Context:
The user is working on a FastAPI project.

Rules:
- Always explain your reasoning.
- Prefer Python 3.10+ idioms.

Response format:
Respond with a code block, then a one-sentence summary.
```

## API

| Method | Description |
|---|---|
| `set_role(text)` | Role section (no heading, order 0) |
| `set_context(text)` | Context section (order 10) |
| `add_rule(rule)` | Append one rule to the rules list (order 20) |
| `set_rules(rules)` | Replace entire rules list |
| `set_format(text)` | Response format section (order 30) |
| `add_section(key, content, *, order, heading)` | Custom section (order 40 by default) |
| `remove_section(key)` | Remove a section by key |
| `has_section(key)` | Check if section exists |
| `build(*, section_sep, list_prefix)` | Render to string |
| `sections()` | All sections in render order |
| `get_section(key)` | Get section by key |
| `clear()` | Remove all sections |
| `to_dict()` / `from_dict(data)` | Serialise/restore |

## License

MIT
