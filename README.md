# agent_runtime

`agent_runtime` is a reusable Python runtime package for agent execution contracts, session planning, service selection, and log lifecycle management.

## Layout

- `src/agent_runtime/` - runtime implementation
- `tests/` - package contract tests
- `context.md` - concise domain context for agents
- `docs/adr/` - architecture decisions that define the runtime boundary

## Development

Install the project in editable mode with dev dependencies, then run the test suite.

```bash
pip install -e ".[dev]"
pytest
```
