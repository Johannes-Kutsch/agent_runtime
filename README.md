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

Ephemeral execution is the canonical entrypoint for an already-rendered prompt when the runtime should not prepare provider-session continuity. Tool access is explicit and defaults to no tools, so callers should pass `ToolAccess.no_tools()` when they want tool-less execution.

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

### New-Session Execution

New-session execution is the canonical entrypoint when the runtime should prepare provider-session continuity and return consumer-owned continuation state for later calls. Tool access is still separate from session lifecycle, so tool-capable runs must pass an explicit `ToolAccess` value.

```python
from pathlib import Path

from agent_runtime import InvocationRole, StageSelection
from agent_runtime.runtime import NewSessionRunRequest, NewSessionRuntime

workspace_tool_access = build_workspace_tool_access()

result = await NewSessionRuntime(
    execution_adapter=build_execution_adapter(),
    service_registry=build_service_registry(),
).run_new_session(
    NewSessionRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        stage=StageSelection(
            service="openai/default",
            model="gpt-5",
            effort="medium",
        ),
        role=InvocationRole("issue-implementation"),
        tool_access=workspace_tool_access,
    )
)

print(result.output)
print(result.continuation)
```

For workspace-backed execution, provide a workspace-backed `ToolAccess` value for the mounted worktree instead of relying on provider defaults. Consumers own persistence for the returned continuation and may store it, discard it, or hand it to a later process.

### Resumed-Session Execution

Resumed-session execution is the canonical entrypoint for continuing an existing provider-session continuity chain. The caller supplies the last continuation, and the runtime resumes without service fallback or tool-policy changes.

```python
from pathlib import Path

from agent_runtime import InvocationRole
from agent_runtime.runtime import ResumedSessionRunRequest, ResumedSessionRuntime

result = await ResumedSessionRuntime(
    execution_adapter=build_execution_adapter(),
    service_registry=build_service_registry(),
).run_resumed_session(
    ResumedSessionRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        role=InvocationRole("issue-implementation"),
        continuation=continuation,
    )
)

print(result.output)
print(result.continuation)
```

Resumed-session execution derives service selection and tool access from the continuation. Callers may continue the chain with a new prompt while keeping the same continuity state.

### Runtime Outcomes

Lifecycle entrypoints report normal runtime outcomes for both completed work and expected interruptions. Completion returns final output and, for session-backed runs, the latest continuation needed for future resumes.

Expected interruptions such as usage limits, cancellation, timeout, temporary service unavailability, and confidently retryable provider failures are canonical runtime outcomes rather than the normal exception path. Session-backed interruption outcomes also report invocation progress so consumers can decide whether to retry immediately or resume from the returned continuation.

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
