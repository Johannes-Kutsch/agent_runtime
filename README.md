# agent_runtime

`agent_runtime` is a reusable Python runtime package for agent execution contracts, session planning, service selection, and log lifecycle management.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`.

```bash
pip install ruhken-agent-runtime
```

## Layout

- `src/agent_runtime/` - runtime implementation
- `tests/` - package contract tests
- `context.md` - concise domain context for agents
- `docs/adr/` - architecture decisions that define the runtime boundary

Execution entrypoints live under `agent_runtime.runtime`; the package root stays narrow and exposes only the stable shared vocabulary.

## Consumer Integration

Ordinary consumers should start with the lifecycle entrypoints under `agent_runtime.runtime` together with the small package-root vocabulary such as `InvocationRole`, `StageSelection`, and `ToolAccess`.

### Ephemeral Execution

Ephemeral execution is the normal path for an already-rendered prompt when the runtime should not prepare provider-session continuity. The caller provides an explicit `InvocationRole` so logs, provider metadata, and state partitioning reflect caller intent.

```python
from pathlib import Path

from agent_runtime import InvocationRole, StageSelection, ToolAccess
from agent_runtime.runtime import EphemeralRunRequest, EphemeralRuntime

runtime = EphemeralRuntime(
    execution_adapter=build_execution_adapter(),
    service_registry=build_service_registry(),
)

result = await runtime.run_ephemeral(
    EphemeralRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        stage=StageSelection(
            service="openai/default",
            model="gpt-5",
            effort="medium",
        ),
        role=InvocationRole("issue-triage"),
        tool_access=ToolAccess.no_tools(),
    )
)

print(result.output)
print(result.selected_service)
```

Use the ephemeral entrypoint when prompt rendering, issue orchestration, and application policy already live in the consuming application and the runtime only needs to execute the prepared prompt without starting a continuity chain.

### Advanced Topics

New-session execution, resumed-session execution, and `ProviderSessionAdapter` integration are advanced seams. Use them when you need provider-backed continuity, recovery, or session-specific policy.

Tool-capable execution is a separate concern from session lifecycle. If an entrypoint can grant tool access, the caller must provide explicit tool access rather than relying on an implicit default. Follow the focused tool-policy documentation for that integration so provider-specific command rendering stays behind adapter contracts.

`ToolPolicyProfile` remains provider-neutral runtime data. Provider adapters translate runtime-owned tool policy values into backend-specific flags, arguments, and restrictions.

## Surface Boundaries

Use these as the normal integration path:

- `agent_runtime.runtime` lifecycle entrypoints for ephemeral, new-session, and resumed-session execution
- Package-root vocabulary such as `InvocationRole` and `StageSelection`

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
