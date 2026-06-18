# agent_runtime

`agent_runtime` is a dependency-free Python runtime package for executing already-prepared agent work through explicit lifecycle entrypoints.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`. Python 3.11 or newer is required.

```bash
pip install ruhken-agent-runtime
```

The package provides the runtime boundary, contracts, service selection, session continuity, and outcome model. It does not ship provider-specific adapters; consuming projects provide adapter wiring or depend on adapter packages built for their provider stack.

For complete signatures, parameters, invariants, and advanced adapter seams, see [the public API reference](docs/public-api.md).

## Consumer Integration

Ordinary consumers should start with lifecycle entrypoints under `agent_runtime.runtime` and the small package-root vocabulary such as `InvocationRole`, `StageSelection`, and `ToolAccess`.

### Ephemeral Execution

Use ephemeral execution for an already-rendered prompt when the runtime should not prepare provider-session continuity. Tool access is explicit; `ToolAccess.no_tools()` is the closed no-tools value.

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
            service="openai-default",
            model="gpt-5",
            effort="medium",
        ),
        role=InvocationRole("issue-triage"),
        tool_access=ToolAccess.no_tools(),
    )
)

if result.kind == "completed":
    print(result.output)
    print(result.selected_service)
```

### New-Session Execution

Use new-session execution when the runtime should prepare provider-session continuity and return consumer-owned continuation state for later calls.

```python
from pathlib import Path

from agent_runtime import InvocationRole, StageSelection, ToolAccess
from agent_runtime.runtime import NewSessionRunRequest, NewSessionRuntime

worktree = Path(".")

result = await NewSessionRuntime(
    execution_adapter=build_execution_adapter(),
    service_registry=build_service_registry(),
).run_new_session(
    NewSessionRunRequest(
        prompt=rendered_prompt,
        worktree=worktree,
        stage=StageSelection(
            service="openai-default",
            model="gpt-5",
            effort="medium",
        ),
        role=InvocationRole("issue-implementation"),
        session_store=build_session_store(),
        provider_session_adapter=build_provider_session_adapter(),
        tool_access=ToolAccess.workspace_backed(worktree),
    )
)

if result.kind == "completed":
    continuation = result.result.continuation
```

Consumers own persistence and retention for returned continuations. The runtime returns the latest continuation; the consuming project decides whether to store it, discard it, or pass it to another process.

### Resumed-Session Execution

Use resumed-session execution to continue an existing provider-session continuity chain. The continuation fixes service selection and tool access; resumed execution does not perform fallback or accept a new tool policy.

```python
from pathlib import Path

from agent_runtime import InvocationRole
from agent_runtime.runtime import (
    ResumedSessionRunRequest,
    ResumedSessionRuntime,
    WorktreeMount,
)

result = await ResumedSessionRuntime(
    execution_adapter=build_execution_adapter(),
).run_resumed_session(
    ResumedSessionRunRequest(
        prompt=rendered_prompt,
        worktree=WorktreeMount(Path(".")),
        role=InvocationRole("issue-implementation"),
        continuation=continuation,
    )
)

if result.kind == "completed":
    continuation = result.result.continuation
```

### Runtime Outcomes

Lifecycle entrypoints return `RuntimeOutcome`. Completed work has `kind == "completed"` and carries the completed result on `result.result`.

Expected interruptions are also normal outcomes: `usage_limited`, `no_service_available`, `cancelled`, `timed_out`, and `retryable_provider_failure`. Session-backed interruption outcomes may carry `continuation` and always report `invocation_progress`, so consumers can choose between retrying the same prompt or continuing the returned provider-session chain.

Exceptional failures remain errors: malformed runtime inputs, credential or configuration problems, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions.
