# agent_runtime Public API

This is the documented `Runtime Public Surface` for `agent_runtime`. It is a target surface for ordinary runtime consumers and maintainers before the first release, not an inventory of every importable symbol in the current implementation.

The distribution name is `ruhken-agent-runtime`; the import package is `agent_runtime`. The package requires Python 3.11 or newer.

The runtime implementation may be split across internal modules. That internal module layout is not part of the `Runtime Public Surface`; consumers should rely on the documented `agent_runtime` and `agent_runtime.runtime` import paths.

## Consumer Surface

Ordinary consumers execute already-rendered prompts through a caller-owned `RuntimeClient`. They provide one `ProviderSelection` per invocation plus tool policy, invocation directory, and session lifecycle data as call arguments. Selection-owned credentials travel on `ProviderSelection` for ephemeral and new-session runs; resumed-session runs accept request-time auth separately because continuations do not store provider secrets. They do not construct provider services, service registries, execution adapters, provider-session adapters, command builders, provider event parsers, or provider DTO streams.

The runtime executes provider work and returns data. Callers own persistence for continuations, observed live events, workflow correlation, durable logs, and usage-limit grouping policy.

### Package Root

Import path: `agent_runtime`

The package root exposes stable shared vocabulary and common errors:

- `AgentCredentialFailureError`
- `AgentRuntimeError`
- `AgentTimeoutError`
- `ClaudeCodeOAuthToken`
- `Continuation`
- `HardAgentError`
- `ProviderAuth`
- `ProviderSelection`
- `ProviderUsage`
- `ResolvedProvider`
- `RuntimeClient`
- `RuntimeConfigurationError`
- `RuntimeOutcome`
- `RunKind`
- `RunResult`
- `ToolPolicy`
- `TransientAgentError`
- `UsageLimitError`

The outcome `kind` discriminator classes are also exported at the package root and from `agent_runtime.runtime`: `Completed`, `UsageLimited`, `NoServiceAvailable`, `Cancelled`, `TimedOut`, and `RetryableProviderFailure`.

The following values and objects are also part of the shared import surface and are available from both `agent_runtime` and `agent_runtime.runtime`:

- `AgentEvent`

Behaviorful lifecycle entrypoints live in `agent_runtime.runtime`, not at the package root.

### Runtime Client

Import path: `agent_runtime.runtime`

```python
RuntimeClient() -> None

async run_ephemeral(request: EphemeralRunRequest) -> RuntimeOutcome
async run_new_session(request: NewSessionRunRequest) -> RuntimeOutcome
async run_resumed_session(request: ResumedSessionRunRequest) -> RuntimeOutcome
```

`run_new_session` implements Start Session Run in public terminology.

`RuntimeClient` does not hold cross-call provider availability or exhaustion policy. It does not own durable provider state, durable logs, process-global auth setup, workflow defaults, prompt rendering, issue orchestration, execution-directory management, dependency installation, or application logging policy. It is safe to reuse across concurrent runtime requests.

### Built-In Providers

Provider selection uses one `ProviderSelection` per runtime invocation. Each selection must reference a built-in service and a model/effort value supported by that built-in integration.

| Service | Auth | Session-backed support | Notes |
| --- | --- | --- | --- |
| `claude` | `ProviderAuth(claude_code_oauth_token=ClaudeCodeOAuthToken(...))` on `ProviderSelection` | Only when the built-in adapter can produce and consume portable continuation data | Providers that cannot satisfy portable continuation requirements are limited to ephemeral execution. |
| `codex` | Host Codex auth files | Only when the built-in adapter can produce and consume portable continuation data | The runtime uses host auth files rather than API-key arguments. |
| `opencode` | `ProviderAuth(opencode_api_key=...)` on `ProviderSelection` | Only when the built-in adapter can produce and consume portable continuation data | Providers that cannot satisfy portable continuation requirements are limited to ephemeral execution. |

The runtime validates built-in service, model, and effort values before provider execution. Invalid service/model/effort references are runtime configuration errors. Missing or invalid explicit credentials are credential failures for that invocation. Runtime outcomes describe the selected provider for one invocation only. Runtime does not perform provider fallback inside a call; consuming projects that want fallback start a separate runtime invocation and own the retry path.

Supported built-in provider allowlists:

**Claude**
- Models: `haiku`, `sonnet`, `opus`
- Efforts: `low`, `medium`, `high`, `xhigh`, `max`

**Codex**
- Models: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`
- Efforts: `low`, `medium`, `high`, `xhigh`

**OpenCode Go** (model list pinned from [opencode.ai](https://opencode.ai))
- Models: `deepseek-v4-flash`, `deepseek-v4-pro`, `glm-5.1`, `glm-5.2`, `kimi-k2.6`, `kimi-k2.7-code`, `mimo-v2.5`, `mimo-v2.5-pro`, `minimax-m2.7`, `minimax-m3`, `qwen3.6-plus`, `qwen3.7-max`, `qwen3.7-plus`
- Efforts: `medium`

`ProviderSelection.model` values for OpenCode Go are service-local identifiers. Provider-specific prefixes such as `opencode-go/` are internal command/config rendering details and are not part of the public API.

### Request Values

Core runtime requests do not require caller-defined labels. Application correlation, workflow naming, display grouping, log naming, and usage-limit grouping belong outside the runtime boundary.

#### `EphemeralRunRequest`

```python
EphemeralRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    provider_selection: ProviderSelection,
    tool_policy: ToolPolicy,
    timeout_seconds: int = 300,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    token: CancellationToken | None = None,
) -> None
```

Ephemeral execution runs an already-rendered prompt without intentionally preparing provider-session continuity. It does not require session storage inputs and does not return a continuation. `timeout_seconds` specifies the idle timeout (maximum seconds without an observed `Agent Event` before runtime aborts); default is 300 seconds.
`on_live_output` can observe `Agent Event` values from this invocation.

#### `NewSessionRunRequest`

```python
NewSessionRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    provider_selection: ProviderSelection,
    tool_policy: ToolPolicy,
    timeout_seconds: int = 300,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    token: CancellationToken | None = None,
) -> None
```

New-session execution selects a built-in provider, starts provider transcript continuity, and returns a portable opaque continuation for later resumed execution. A completed session-backed run must include meaningful continuation data; if a provider cannot produce resume data, the runtime must not report a successful session-backed result. `timeout_seconds` specifies the idle timeout (maximum seconds without an observed `Agent Event` before runtime aborts); default is 300 seconds.
`on_live_output` is also available for Start Session Run.

#### `ResumedSessionRunRequest`

```python
ResumedSessionRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    continuation: Continuation,
    provider_auth: ProviderAuth | None = None,
    timeout_seconds: int = 300,
    on_live_output: Callable[[AgentEvent], None] | None = None,
    token: CancellationToken | None = None,
) -> None
```

Resumed-session execution continues an existing provider-session continuity chain. The continuation fixes service, model, effort, and tool policy. Resumed execution accepts request-time credentials because continuations do not store provider secrets, but it does not perform fallback, reselection, or tool-policy replacement. `timeout_seconds` specifies the idle timeout (maximum seconds without an observed `Agent Event` before runtime aborts); default is 300 seconds.
`on_live_output` is also available for Resume Session Run.

### Live Runtime Output

`on_live_output` publishes per-invocation `Agent Event` observations while a run is executing. Observers receive events from the current invocation only.

`Agent Event` values are discriminated by type: `agent_message` (assistant message output), `agent_tool_call` (tool invocation), `turn_summary` (provider-emitted turn/step completion carrying usage and cost data), or `other` (agent life sign). Each event carries both a `display_message` and the raw provider output it derived from.

- Live Runtime Output is `Agent Event` observation, not token-by-token streaming.
- Events carry both a filtered `display_message` and raw provider output for consumers to choose which representation to use.
- It does not replay prior events from continuations, logs, or historical transcript state.
- Completed runtime output and `RuntimeOutcome` interruption semantics remain authoritative and unchanged.
- `on_live_output` callbacks are synchronous and notification-only. Callback exceptions are propagated to the caller as consumer failures.
- Consumers own backpressure policy, queueing, display formatting, redaction, and persistence for observed events.
- Live Runtime Output observers are for display/telemetry only and do not control runtime flow.

### Auth

#### `ProviderAuth`

```python
ProviderAuth(
    *,
    claude_code_oauth_token: ClaudeCodeOAuthToken | str | None = None,
    opencode_api_key: str | None = None,
) -> None
```

`ProviderAuth` is immutable credential data carried by `ProviderSelection` for a new provider selection, or supplied directly to Resume Session Run for credentials required by the continued provider. It only needs credentials for the selected explicit-credential provider. Codex uses host auth files. Continuations must not store provider credentials.

### Outcomes and Continuations

#### `RuntimeOutcome`

```python
RuntimeOutcome(
    kind: Completed | UsageLimited | NoServiceAvailable | Cancelled | TimedOut | RetryableProviderFailure,
    result: RunResult,
) -> None
```

Lifecycle entrypoints return `RuntimeOutcome` for both completed work and expected interruption outcomes. `kind` is a variant object that both discriminates the outcome and carries any kind-specific data; `result` is always present and exposes the run's output, usage, continuation, and selected provider. Each `RuntimeOutcome` describes exactly one invocation and never aggregates fallback attempts across providers.

Consumers branch on `kind` with `isinstance`:

```python
match outcome.kind:
    case Completed():
        ...
    case UsageLimited(reset_time=reset):
        ...
```

Outcome kinds:

- `Completed()`: work completed; `result.output` is the final output and `result.continuation` is present for session-backed runs.
- `UsageLimited(reset_time: datetime | None)`: usage limit interrupted execution.
- `NoServiceAvailable(reset_time: datetime | None)`: the selected provider is temporarily unavailable before model work starts.
- `Cancelled()`: caller- or user-initiated cancellation.
- `TimedOut()`: runtime timeout. For OpenCode Go, this outcome is ambiguous by design: from the runtime's side, an exhausted subscription quota and a server-side maintenance outage are indistinguishable. Consumers should treat `TimedOut()` as back-off / `Consumer Fallback` eligible, not as proof of exhausted quota. The runtime does not distinguish the cause and does not perform any pre-flight subscription or usage check.
- `RetryableProviderFailure()`: provider failure classified as confidently retryable.

Expected interruption outcomes are normal lifecycle results. Credential failures, malformed inputs, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions remain errors.

Every outcome's `result` exposes the selected provider (`result.selected`), any provider usage observed before completion or interruption, and continuation state when relevant. Reset-time facts live on the `UsageLimited`/`NoServiceAvailable` kind objects. Caller-defined usage-limit grouping is not part of the core runtime API.

#### `RunResult`

```python
RunResult(
    *,
    output: str,
    usage: ProviderUsage | None,
    continuation: Continuation | None,
    selected: ResolvedProvider,
) -> None
```

`RunResult` is the single result shape for every outcome kind. `output` is the run's text output (empty for pre-output interruptions). `usage` is provider usage when reported. `continuation` is the latest resume token for session-backed runs, or `None` for ephemeral runs and pre-start interruptions. `selected` identifies the provider actually run.

#### `ResolvedProvider`

```python
ResolvedProvider(
    *,
    service: str,
    model: str,
    effort: str,
) -> None
```

Credential-free identity of the provider that ran the invocation. It carries no auth, continuation, or session data — only the selected service, model, and effort.

#### `Continuation`

```python
Continuation(serialized: str) -> None
```

`Continuation` is an opaque portable resume token. Callers may persist and pass it back to runtime entrypoints, but they must not inspect or depend on provider resume payloads as a stable schema.

The runtime may carry provider-owned resume data inside the continuation, including encoded provider state when needed. Provider credentials do not belong in continuations.

The runtime does not return structured finished-run records. Callers that want to persist or display execution traces observe `Agent Event` values through `on_live_output` during the run and own their persistence, file layout, naming, redaction, retention, and cleanup.

#### `ProviderUsage`

```python
ProviderUsage(
    *,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cache_read_input_tokens: int | None = None,
    cache_creation_input_tokens: int | None = None,
    cost_usd: Decimal | None = None,
    duration_seconds: float | None = None,
) -> None
```

Provider usage is top-level optional outcome metadata for completed and interrupted outcomes when the provider reports it. Built-in provider parsers emit rich usage events; `PromptTokens` is not the usage contract. Cancellation and timeout outcomes report only usage observed before interruption.

### Runtime Value Objects

#### `ProviderSelection`

```python
ProviderSelection(
    *,
    service: str,
    model: str,
    effort: str,
    auth: ProviderAuth | None = None,
) -> None
```

One built-in provider candidate for one runtime invocation, including credentials required by that provider. Construction validates value shape; invocation validates built-in provider support, relevant credentials, and availability. Consumer-owned fallback means callers choose a different `ProviderSelection` only by starting a later invocation.

#### `ToolPolicy`

```python
ToolPolicy.NONE
ToolPolicy.INSPECT_ONLY
ToolPolicy.NO_FILE_MUTATION
ToolPolicy.UNRESTRICTED
```

Closed runtime value for provider tool access. `ToolPolicy` is the consumer-facing tool contract. Provider flag profiles, MCP config rendering, and command-line policy mappings are internal Built-in Provider Adapter policy.

Every runtime request carries an Invocation Directory through `invocation_dir`; it is the host directory where the provider command is launched. `ToolPolicy.NONE` forbids provider tools. Any non-`NONE` policy grants provider tools against that Invocation Directory, making it the Tool Workspace without requiring a second public path.

Policy meanings:

- `NONE`: provider tools are forbidden.
- `INSPECT_ONLY`: workspace inspection tools are allowed.
- `NO_FILE_MUTATION`: tools may be available, but direct workspace file mutation is forbidden.
- `UNRESTRICTED`: the runtime adds no tool restriction beyond provider defaults.

Every public `ToolPolicy` is valid for every supported service/model pair. Built-in Provider Adapters enforce each policy with the strongest provider-supported mechanism, so enforcement strength may vary by provider. A provider-session continuity chain keeps the same `ToolPolicy` for its lifetime.

#### `AgentEvent`

```python
AgentEvent(
    *,
    type: Literal["agent_message", "agent_tool_call", "turn_summary", "other"],
    display_message: str,
    raw_provider_output: str,
) -> None
```

One observed signal from Live Runtime Output, discriminated by type:

- `agent_message` — assistant text output. `display_message` is the message text.
- `agent_tool_call` — tool invocation. `display_message` is a rendered `tool_name(payload)` string.
- `turn_summary` — provider-emitted turn or step completion event carrying usage and cost data (Claude `result`, Codex `turn.completed`, OpenCode `step_finish`). `display_message` summarises available fields such as stop reason, duration, cost, and token counts.
- `other` — agent life sign or provider diagnostic. `display_message` is a descriptor drawn from the provider event type, subtype, or relevant fields (e.g. `system.init cwd=<path>`, `system.thinking_tokens tokens=<n>`), or the stripped raw text for non-JSON provider output.

Each event carries `display_message` alongside `raw_provider_output`. Consumers choose which representation to use.

The type is shared runtime vocabulary and is importable from both `agent_runtime` and `agent_runtime.runtime`.

## Adapter Surface

There is no supported consumer-defined provider adapter surface in the target runtime. Provider services, service registries, execution adapters, provider-session adapters, provider output DTOs, provider state capture/restore details, command builders, and provider event parsers are runtime-owned internals.

Public provider failure errors may expose `ProviderErrorObservation` for diagnostics. Consumers own storage, display, and redaction policy for those diagnostics.
