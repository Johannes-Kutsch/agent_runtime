# agent_runtime

`agent_runtime` is the reusable Python runtime package for executing already-prepared agent work through built-in provider integrations.

Install the distribution as `ruhken-agent-runtime` and import it as `agent_runtime`. Python 3.11 or newer is required.

```bash
pip install ruhken-agent-runtime
```

The accepted runtime direction is to ship Claude, Codex, and OpenCode execution inside this package. Consuming projects select a built-in provider, model, effort, credentials, tool policy, invocation directory, and session lifecycle through runtime call arguments; they do not construct provider services, service registries, command builders, provider-session adapters, or provider event parsers.

For complete target signatures and invariants, see [the public API reference](docs/public-api.md). For the portable continuation decision, see [ADR 0005](docs/adr/0005-runtime-session-lifecycle-entrypoints.md).

Only the documented import paths are stable. Internal runtime modules may be reorganized as the implementation is split, but ordinary consumers should continue importing from `agent_runtime` and `agent_runtime.runtime`.

## Consumer Integration

Ordinary consumers should use a caller-owned `RuntimeClient` and the small package vocabulary such as `StageSelection`, `ToolPolicy`, `ProviderAuth`, and `Continuation`.

The runtime executes prompts and returns data. Callers own persistence for continuations, invocation records, workflow correlation, durable logs, and any usage-limit grouping policy.

Every run receives an `invocation_dir`, the host directory where the provider command is launched. Tool policy is explicit: `ToolPolicy.NONE` forbids provider tools, `ToolPolicy.INSPECT_ONLY` allows workspace inspection, `ToolPolicy.NO_FILE_MUTATION` permits tools while forbidding direct workspace file mutation, and `ToolPolicy.UNRESTRICTED` adds no runtime restriction beyond provider defaults.

### Ephemeral Execution

Use ephemeral execution for an already-rendered prompt when the runtime should not prepare provider-session continuity. Tool policy is explicit; `ToolPolicy.NONE` is the closed no-tools value.

```python
from pathlib import Path

from agent_runtime import ProviderAuth, StageSelection, ToolPolicy
from agent_runtime.runtime import EphemeralRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_ephemeral(
    EphemeralRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        stage=StageSelection(
            service="claude",
            model="sonnet",
            effort="medium",
        ),
        provider_auth=ProviderAuth(
            claude_code_oauth_token=claude_code_oauth_token,
        ),
        tool_policy=ToolPolicy.NONE,
    )
)

if result.kind == "completed":
    print(result.output)
    print(result.usage)
```

Ephemeral execution does not return a continuation and does not require session storage inputs.

### New-Session Execution

Use new-session execution when the runtime should preserve provider transcript continuity and return an opaque portable `Continuation` for later calls. A completed session-backed run always returns output text and a meaningful continuation.

```python
from pathlib import Path

from agent_runtime import ProviderAuth, StageSelection, ToolPolicy
from agent_runtime.runtime import NewSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_new_session(
    NewSessionRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        stage=StageSelection(
            service="opencode",
            model="github-copilot/gpt-5.1-codex",
            effort="medium",
        ),
        provider_auth=ProviderAuth(opencode_api_key=opencode_api_key),
        tool_policy=ToolPolicy.NO_FILE_MUTATION,
    )
)

if result.kind == "completed":
    print(result.output)
    continuation = result.result.continuation
```

Callers persist the continuation object wherever they want. The continuation is a resume token, not a public schema for provider state, display data, or policy decisions.

### Resumed-Session Execution

Use resumed-session execution to continue an existing provider-session continuity chain. The continuation fixes the selected service and tool policy. Resumed execution does not perform fallback and only allows model or effort overrides.

```python
from pathlib import Path

from agent_runtime import ProviderAuth
from agent_runtime.runtime import ResumedSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_resumed_session(
    ResumedSessionRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        continuation=continuation,
        provider_auth=ProviderAuth(opencode_api_key=opencode_api_key),
    )
)

if result.kind == "completed":
    print(result.output)
    continuation = result.result.continuation
```

### Invocation Records

The runtime may return structured invocation records for callers that want traces. Callers decide if, where, and how to persist those records. The runtime does not own durable log file names, directories, retention, or cleanup policy.

### Runtime Outcomes

Lifecycle entrypoints return `RuntimeOutcome`. Completed work has `kind == "completed"` and carries the completed result on `result.result`. When a provider reports usage, the outcome also carries `usage` with input tokens, output tokens, cache-read input tokens, cache-creation input tokens, optional cost, and optional provider duration.

Expected interruptions are normal outcomes: `usage_limited`, `no_service_available`, `cancelled`, `timed_out`, and `retryable_provider_failure`. Session-backed interruption outcomes may carry `continuation` only when provider progress made resume meaningful, and they always report `invocation_progress`.

Usage-limit outcomes expose provider and service facts such as service name, account label, reset time, invocation progress, provider usage, and continuation state. Caller workflow grouping and retry/sleep policy stay outside the runtime package.

Exceptional failures remain errors: malformed runtime inputs, credential problems, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions.
