# agent_runtime

`agent_runtime` is a reusable Python runtime package for agent execution contracts, session planning, service selection, and log lifecycle management.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`.

## Layout

- `src/agent_runtime/` - runtime implementation
- `tests/` - package contract tests
- `context.md` - concise domain context for agents
- `docs/adr/` - architecture decisions that define the runtime boundary

Execution entrypoints live under `agent_runtime.runtime`; the package root is a narrow compatibility entrypoint for the stable runtime surface.

## Consumer Integration

Ordinary consumers should use the canonical runtime entrypoints under `agent_runtime.runtime` together with the small package-root vocabulary such as `InvocationRole`, `UsageLimitScope`, `StageSelection`, and `ExecutionProvider`.

### One-shot Execution

One-shot execution is the normal path for an already-rendered prompt. The caller provides an explicit `InvocationRole` so logs, provider metadata, and state partitioning reflect caller intent instead of an implicit runtime default.

```python
from agent_runtime import ExecutionProvider, InvocationRole, StageSelection
from agent_runtime.runtime import one_shot

provider: ExecutionProvider = build_provider()
selection = StageSelection(service="openai/default", model="gpt-5", effort="medium")

result = await one_shot(
    prompt=rendered_prompt,
    invocation_role=InvocationRole("issue-triage"),
    stage_selection=selection,
    provider=provider,
)

print(result.output_text)
print(result.selected_service)
```

Use the one-shot entrypoint when prompt rendering, issue orchestration, and application policy already live in the consuming application and the runtime only needs to execute the prepared prompt.

### Resumable Execution

The provider-session-backed mode is the resumable path, which is the post-refactor public name for the older resident execution concept. The invocation role should come from session planning and then flow unchanged into execution.

```python
from agent_runtime import InvocationRole, UsageLimitScope
from agent_runtime.runtime import resumable
from agent_runtime.runtime.sessions import plan_session

plan = await plan_session(
    invocation_role=InvocationRole("implementer"),
    usage_limit_scope=UsageLimitScope("repo-write"),
    provider_session_adapter=session_adapter,
    provider=provider,
)

result = await resumable(plan=plan, prompt=rendered_prompt)
print(result.output_text)
```

`UsageLimitScope` is optional. Use it only when quota policy should be grouped differently from invocation identity. If it is omitted, ordinary consumers can let usage-limit policy follow the `InvocationRole` instead of introducing a second grouping key.

## Surface Boundaries

Use these as the normal integration path:

- `agent_runtime.runtime` canonical entrypoints for one-shot and resumable execution
- Package-root vocabulary such as `InvocationRole`, `UsageLimitScope`, `StageSelection`, and `ExecutionProvider`

Use these only when you are building or extending adapters:

- Focused session-planning and provider-session seams
- Provider-policy and provider-specific DTO modules
- Provider-session path or recovery helpers

Do not treat low-level work invocation modules as the ordinary consumer API. Direct `invoke_work` usage is an advanced implementation seam, not the normal way to integrate a runtime consumer.

## Development

Install the project in editable mode with dev dependencies, then run the test suite.

```bash
pip install -e ".[dev]"
pytest
```
