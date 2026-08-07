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

Ordinary consumers should use a caller-owned `RuntimeClient` and the small package vocabulary such as `ProviderSelection`, `ToolPolicy`, `ProviderAuth`, and `Continuation`.

The runtime executes prompts and returns data. Callers own persistence for continuations, live output observations, workflow correlation, durable logs, and any usage-limit grouping policy.

Every run receives an `invocation_dir`, the host directory where the provider command is launched. Tool policy is explicit: `ToolPolicy.NONE` forbids provider tools, `ToolPolicy.NO_FILE_MUTATION` permits tools while forbidding direct workspace file mutation, and `ToolPolicy.UNRESTRICTED` adds no runtime restriction beyond provider defaults.

### Ephemeral Execution

Use ephemeral execution for an already-rendered prompt when the runtime should not prepare provider-session continuity. Tool policy is explicit; `ToolPolicy.NONE` is the closed no-tools value.

```python
from pathlib import Path

from agent_runtime import Completed, ProviderAuth, ProviderSelection, ToolPolicy
from agent_runtime.runtime import EphemeralRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_ephemeral(
    EphemeralRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        provider_selection=ProviderSelection(
            service="claude",
            model="sonnet",
            effort="medium",
            auth=ProviderAuth(
                claude_code_oauth_token=claude_code_oauth_token,
            ),
        ),
        tool_policy=ToolPolicy.NONE,
    )
)

if isinstance(result.kind, Completed):
    print(result.result.output)
    print(result.result.usage)
```

Ephemeral execution does not return a continuation and does not require session storage inputs.

### New-Session Execution

Use new-session execution when the runtime should preserve provider transcript continuity and return an opaque portable `Continuation` for later calls. A completed session-backed run always returns output text and a meaningful continuation.

```python
from pathlib import Path

from agent_runtime import Completed, ProviderAuth, ProviderSelection, ToolPolicy
from agent_runtime.runtime import NewSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_new_session(
    NewSessionRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        provider_selection=ProviderSelection(
            service="opencode",
            model="deepseek-v4-flash",
            effort="medium",
            auth=ProviderAuth(opencode_api_key=opencode_api_key),
        ),
        tool_policy=ToolPolicy.NO_FILE_MUTATION,
        session_store=Path("./sessions"),
    )
)

if isinstance(result.kind, Completed):
    print(result.result.output)
    continuation = result.result.continuation
```

Callers persist the continuation object wherever they want. The continuation is a resume token, not a public schema for provider state, display data, or policy decisions.

### Resumed-Session Execution

Use resumed-session execution to continue an existing provider-session continuity chain. The continuation fixes the selected service and tool policy. Resumed execution does not perform fallback and only allows model or effort overrides.

```python
from pathlib import Path

from agent_runtime import Completed, ProviderAuth
from agent_runtime.runtime import ResumedSessionRunRequest, RuntimeClient

runtime = RuntimeClient()

result = await runtime.run_resumed_session(
    ResumedSessionRunRequest(
        prompt=rendered_prompt,
        invocation_dir=Path("."),
        continuation=continuation,
        provider_auth=ProviderAuth(opencode_api_key=opencode_api_key),
        session_store=Path("./sessions"),
    )
)

if isinstance(result.kind, Completed):
    print(result.result.output)
    continuation = result.result.continuation
```

### Live Output

All run requests accept an optional `on_live_output: Callable[[AgentEvent], None]` callback. The runtime calls it synchronously for each `AgentEvent` observed during the run. `AgentEvent` values carry a `type` (`"agent_message"`, `"agent_tool_call"`, `"turn_summary"`, or `"other"`), a `display_message`, and `raw_provider_output`.

Live output is notification-only and does not control runtime flow. Callbacks must not raise; exceptions propagate to the caller as consumer failures. The runtime does not replay prior events from continuations or history. Consumers own buffering, display formatting, persistence, and redaction for observed events.

### Runtime Outcomes

Lifecycle entrypoints return `RuntimeOutcome`, whose `kind` is one of a closed set of outcome values: `Completed`, `UsageLimited`, `ProviderUnavailable`, `ModelNotAvailable`, `Cancelled`, `TimedOut`. Discriminate with `isinstance(outcome.kind, Completed)` — `kind` is a value object, not a string. Completed work carries its output on `outcome.result.output`. When a provider reports usage, `outcome.result.usage` carries input tokens, output tokens, cache-read input tokens, cache-creation input tokens, optional cost, and optional provider duration.

Expected interruptions are normal outcomes rather than exceptions: `UsageLimited`, `ProviderUnavailable` (carrying a closed `reason` of `TRANSIENT_API_ERROR` or `SERVICE_NOT_AVAILABLE`), `ModelNotAvailable`, `Cancelled`, and `TimedOut`. Session-backed interruption outcomes may carry a continuation on `result.continuation` only when provider progress made resume meaningful.

`UsageLimited` carries `reset_time` (when the limit resets, or `None` if unknown) and `is_permanent`. Service identity and provider usage are available on `result.selected` and `result.usage` as with all outcomes. `is_permanent=True` signals that the account is permanently exhausted rather than temporarily rate-limited; consumers use it to decide whether to schedule a retry or mark an account unavailable. Caller workflow grouping and retry/sleep policy stay outside the runtime package.

## Custom Provider Execution

By default, `RuntimeClient` spawns provider CLIs as host subprocesses. A `ProviderInvocationAdapter` replaces that execution step without requiring a Docker or container dependency on `agent_runtime`. Provide one when the consuming project needs to route provider execution to a non-host environment — for example, a container, a remote host, or a test double.

The adapter is infrastructure stable for the lifetime of the client: inject it once at construction, and all three run entry-points (`run_ephemeral`, `run_new_session`, `run_resumed_session`) thread it through automatically.

### Implementing the Protocol

Import everything from `agent_runtime`:

```python
from agent_runtime import (
    InvocationFailureKind,
    ProviderInvocationAdapter,
    ProviderInvocationFailure,
    ProviderInvocationRequest,
    ProviderInvocationResult,
    consume_provider_stdout_lines,
)
```

A minimal adapter that delegates to a remote executor:

```python
class RemoteProviderAdapter:
    def execute(
        self,
        request: ProviderInvocationRequest,
        argv_transform=None,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        lines = self._remote_run(request.argv, request.worktree, request.environment)
        output, usage = request.output_hooks.reduce_output(list(lines))
        return ProviderInvocationResult(output=output, usage=usage, stdout_lines=tuple(lines))

    def _remote_run(self, argv, worktree, env) -> list[str]:
        ...
```

`request.worktree` is the Invocation Directory. `request.output_hooks.reduce_output` is the stream interpreter; call it with all collected stdout lines to get the final output string and optional `ProviderUsage`.

### Injecting the Adapter

Pass the adapter to `RuntimeClient` at construction:

```python
from agent_runtime.runtime import RuntimeClient

adapter = RemoteProviderAdapter()
runtime = RuntimeClient(provider_invocation_adapter=adapter)

# All three run kinds use the adapter automatically:
result = await runtime.run_ephemeral(ephemeral_request)
result = await runtime.run_new_session(new_session_request)
result = await runtime.run_resumed_session(resumed_session_request)
```

### Returning a Classified Failure

Return `ProviderInvocationFailure` instead of raising when the remote executor reports a recognised failure. Use `InvocationFailureKind` to classify it; the runtime turns the failure into the appropriate `RuntimeOutcome` (`UsageLimited` or `ProviderUnavailable`):

```python
from datetime import datetime

# Usage limit — temporary rate limit with a known reset time:
return ProviderInvocationFailure(
    kind=InvocationFailureKind.USAGE_LIMITED,
    detail="rate limit exceeded",
    reset_time=datetime(2026, 8, 8, 0, 0, 0),
    is_permanent=False,
)

# Usage limit — permanent account exhaustion:
return ProviderInvocationFailure(
    kind=InvocationFailureKind.USAGE_LIMITED,
    detail="account permanently exhausted",
    is_permanent=True,
)

# Transient provider unavailability:
return ProviderInvocationFailure(
    kind=InvocationFailureKind.PROVIDER_UNAVAILABLE,
    detail="upstream 503",
)
```

### Streaming Live Runtime Output Incrementally

Adapters that receive output line-by-line should call `consume_provider_stdout_lines` per batch so that `Live Runtime Output` is delivered incrementally rather than after the full run completes. The call is a no-op when the stream interpreter does not support incremental delivery, so it is always safe:

```python
class StreamingRemoteAdapter:
    def execute(
        self,
        request: ProviderInvocationRequest,
        argv_transform=None,
    ) -> ProviderInvocationResult | ProviderInvocationFailure:
        all_lines: list[str] = []
        for line in self._stream_from_remote(request.argv, request.worktree):
            all_lines.append(line)
            consume_provider_stdout_lines(request.output_hooks.reduce_output, [line])
        output, usage = request.output_hooks.reduce_output(all_lines)
        return ProviderInvocationResult(
            output=output,
            usage=usage,
            stdout_lines=tuple(all_lines),
        )

    def _stream_from_remote(self, argv, worktree):
        ...
```

#### Retryable versus hard provider failures

A provider failure the runtime judges temporary is **returned**, never raised: server-side 5xx responses, and any failure a service's classifier recognises as transient, arrive as a `ProviderUnavailable` outcome with `reason=TRANSIENT_API_ERROR`. Retrying is your decision — the runtime never waits, retries, or falls back on its own.

A provider failure judged permanent **raises** `HardAgentError`: provider-reported 4xx-class failures, process-level failures (non-zero exit, empty output), and failures a service's classifier cannot identify. Discriminate hard failures by exception type — `AgentCredentialFailureError` is the credential-specific subclass — not by the `classification` field, which is populated only for credential failures and is `None` on a plain `HardAgentError`. Provider HTTP status codes are deliberately not propagated onto exceptions. Exception: OpenCode's `401 invalid api key` signals permanent account exhaustion rather than misconfiguration; the runtime surfaces it as `UsageLimited(is_permanent=True)` rather than raising `AgentCredentialFailureError`.

Which signals a given service treats as transient is per-service knowledge and may differ between Claude, Codex, and OpenCode.

Other exceptional failures remain errors: malformed runtime inputs, most credential problems, adapter or protocol bugs, and unexpected exceptions.
