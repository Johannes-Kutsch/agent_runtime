# agent_runtime

`agent_runtime` is the reusable Python runtime package for executing already-prepared agent work through built-in provider integrations.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`. Python 3.11 or newer is required.

```bash
pip install ruhken-agent-runtime
```

The accepted runtime direction is to ship Claude, Codex, and OpenCode execution inside this package. Consuming projects select a built-in provider, model, effort, credentials, tool access, and session lifecycle through runtime call arguments; they do not construct provider services, service registries, command builders, provider-session adapters, or provider event parsers.

For complete target signatures, invariants, and migration notes, see [the public API reference](docs/public-api.md).

Only the documented import paths are stable. Internal runtime modules may be reorganized as the implementation is split, but ordinary consumers should continue importing from `agent_runtime` and `agent_runtime.runtime`.

## Consumer Integration

Ordinary consumers should use a caller-owned `RuntimeClient` and the small package vocabulary such as `InvocationRole`, `StageSelection`, `ToolAccess`, and `ProviderAuth`.

### Ephemeral Execution

Use ephemeral execution for an already-rendered prompt when the runtime should not prepare provider-session continuity. Tool access is explicit; `ToolAccess.no_tools()` is the closed no-tools value.

```python
from pathlib import Path

from agent_runtime import InvocationRole, ProviderAuth, StageSelection, ToolAccess
from agent_runtime.runtime import EphemeralRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_ephemeral(
    EphemeralRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        stage=StageSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        role=InvocationRole("issue-triage"),
        provider_auth=ProviderAuth(
            claude_code_oauth_token=claude_code_oauth_token,
        ),
        tool_access=ToolAccess.no_tools(),
    )
)

if result.kind == "completed":
    print(result.output)
    print(result.usage)
```

Older adapter-wired runtime wrappers remain internal compatibility seams and are not part of the documented `Runtime Public Surface`.

### New-Session Execution

Use new-session execution when the runtime should preserve provider-native transcript continuity and return consumer-owned continuation state for later calls. Session-backed calls require an explicit `runtime_state_dir`; durable invocation logs remain opt-in through `logs_dir`.

```python
from pathlib import Path

from agent_runtime import InvocationRole, ProviderAuth, StageSelection, ToolAccess
from agent_runtime.runtime import NewSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_new_session(
    NewSessionRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        runtime_state_dir=Path(".agent-runtime/state"),
        stage=StageSelection(
            service="opencode",
            model="github-copilot/gpt-5.1-codex",
            effort="medium",
        ),
        role=InvocationRole("issue-implementation"),
        provider_auth=ProviderAuth(opencode_api_key=opencode_api_key),
        tool_access=ToolAccess.workspace_backed(Path(".")),
    )
)

if result.kind == "completed":
    continuation = result.result.continuation
```

Consumers own persistence and retention for returned continuations and for `runtime_state_dir`. The runtime returns the latest continuation; the consuming project decides whether to store it, discard it, or pass it to another process.

### Resumed-Session Execution

Use resumed-session execution to continue an existing provider-session continuity chain. The continuation fixes the selected service and tool access; resumed execution does not perform fallback and only allows model or effort overrides.

```python
from pathlib import Path

from agent_runtime import InvocationRole, ProviderAuth
from agent_runtime.runtime import ResumedSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_resumed_session(
    ResumedSessionRunRequest(
        prompt=rendered_prompt,
        worktree=Path("."),
        runtime_state_dir=Path(".agent-runtime/state"),
        role=InvocationRole("issue-implementation"),
        continuation=continuation,
        provider_auth=ProviderAuth(opencode_api_key=opencode_api_key),
    )
)

if result.kind == "completed":
    continuation = result.result.continuation
```

Older resumed-session wrapper spellings remain internal compatibility seams and are not the intended consumer integration path.

### Runtime Outcomes

Lifecycle entrypoints return `RuntimeOutcome`. Completed work has `kind == "completed"` and carries the completed result on `result.result`. When a provider reports usage, the outcome also carries `usage` with input tokens, output tokens, cache-read input tokens, cache-creation input tokens, optional cost, and optional provider duration.

Expected interruptions are normal outcomes: `usage_limited`, `no_service_available`, `cancelled`, `timed_out`, and `retryable_provider_failure`. Session-backed interruption outcomes may carry `continuation` and always report `invocation_progress`, so consumers can choose between retrying the same prompt or continuing the returned provider-session chain.

Exceptional failures remain errors: malformed runtime inputs, credential problems, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions.
