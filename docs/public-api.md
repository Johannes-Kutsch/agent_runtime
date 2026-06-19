# agent_runtime Public API

This is the documented `Runtime Public Surface` for `agent_runtime`. It is a target surface for ordinary runtime consumers and maintainers before the first release, not an inventory of every importable symbol in the current implementation.

The distribution name is `ruhken-agent-runtime`; the import package is `agent_runtime`. The package requires Python 3.11 or newer.

The runtime implementation may be split across internal modules. That internal module layout is not part of the `Runtime Public Surface`; consumers should rely on the documented `agent_runtime` and `agent_runtime.runtime` import paths.

## Consumer Surface

Ordinary consumers execute already-rendered prompts through a caller-owned `RuntimeClient`. They provide provider selection, credentials, tool policy, invocation directory, and session lifecycle data as call arguments. They do not construct provider services, service registries, execution adapters, provider-session adapters, command builders, provider event parsers, or provider DTO streams.

The runtime executes provider work and returns data. Callers own persistence for continuations, invocation records, workflow correlation, durable logs, and usage-limit grouping policy.

### Package Root

Import path: `agent_runtime`

The package root exposes stable shared vocabulary and common errors:

- `AgentCredentialFailureError`
- `AgentFailedError`
- `AgentRuntimeError`
- `AgentTimeoutError`
- `ClaudeCodeOAuthToken`
- `Continuation`
- `HardAgentError`
- `InvocationProgress`
- `InvocationRecord`
- `ProviderAuth`
- `ProviderUsage`
- `RuntimeConfigurationError`
- `RuntimeOutcome`
- `RunKind`
- `StageSelection`
- `ToolPolicy`
- `TransientAgentError`
- `UsageLimitError`

Behaviorful lifecycle entrypoints live in `agent_runtime.runtime`, not at the package root.

### Runtime Client

Import path: `agent_runtime.runtime`

```python
RuntimeClient() -> None

async run_ephemeral(request: EphemeralRunRequest) -> RuntimeOutcome
async run_new_session(request: NewSessionRunRequest) -> RuntimeOutcome
async run_resumed_session(request: ResumedSessionRunRequest) -> RuntimeOutcome
```

`RuntimeClient` holds in-process built-in provider availability and exhaustion state across calls. It does not own durable provider state, durable logs, process-global auth setup, workflow defaults, prompt rendering, issue orchestration, managed worktrees, dependency installation, or application logging policy. It is safe to reuse across concurrent runtime requests and synchronizes provider availability updates internally.

### Built-In Providers

Provider selection uses `StageSelection` nodes. Every selected node must reference a built-in service and a model/effort value supported by that built-in integration.

| Service | Auth | Session-backed support | Notes |
| --- | --- | --- | --- |
| `claude` | `ProviderAuth(claude_code_oauth_token=ClaudeCodeOAuthToken(...))` | Only when the built-in adapter can produce and consume portable continuation data | Providers that cannot satisfy portable continuation requirements are limited to ephemeral execution. |
| `codex` | Host Codex auth files | Only when the built-in adapter can produce and consume portable continuation data | The runtime uses host auth files rather than API-key arguments. |
| `opencode` | `ProviderAuth(opencode_api_key=...)` | Only when the built-in adapter can produce and consume portable continuation data | Providers that cannot satisfy portable continuation requirements are limited to ephemeral execution. |

The runtime validates built-in service, model, and effort values before provider execution. Invalid service/model/effort references are runtime configuration errors. Missing or invalid explicit credentials are credential failures and stop execution rather than triggering fallback.

Known migrated pycastle allowlists:

- Claude models: `haiku`, `sonnet`, `opus`; efforts: `low`, `medium`, `high`, `xhigh`, `max`.
- Codex models: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`; efforts: `low`, `medium`, `high`, `xhigh`.
- OpenCode models and command mappings come from pycastle's OpenCode service; effort is currently `medium`.

### Request Values

Core runtime requests do not require caller-defined labels. Application correlation, workflow naming, display grouping, log naming, and usage-limit grouping belong outside the runtime boundary.

#### `EphemeralRunRequest`

```python
EphemeralRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    stage: StageSelection,
    provider_auth: ProviderAuth,
    tool_policy: ToolPolicy,
    token: CancellationToken | None = None,
) -> None
```

Ephemeral execution runs an already-rendered prompt without intentionally preparing provider-session continuity. It does not require session storage inputs and does not return a continuation.

#### `NewSessionRunRequest`

```python
NewSessionRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    stage: StageSelection,
    provider_auth: ProviderAuth,
    tool_policy: ToolPolicy,
    token: CancellationToken | None = None,
) -> None
```

New-session execution selects a built-in provider, starts provider transcript continuity, and returns a portable opaque continuation for later resumed execution. A completed session-backed run must include meaningful continuation data; if a provider cannot produce resume data, the runtime must not report a successful session-backed result.

#### `ResumedSessionRunRequest`

```python
ResumedSessionRunRequest(
    *,
    prompt: str,
    invocation_dir: Path,
    continuation: Continuation,
    provider_auth: ProviderAuth,
    model: str | None = None,
    effort: str | None = None,
    token: CancellationToken | None = None,
) -> None
```

Resumed-session execution continues an existing provider-session continuity chain. The continuation fixes service and tool policy. Resumed execution does not perform fallback and rejects tool-policy replacement; omitted model or effort values default from result metadata associated with the continuation when available.

### Auth

#### `ProviderAuth`

```python
ProviderAuth(
    *,
    claude_code_oauth_token: ClaudeCodeOAuthToken | str | None = None,
    opencode_api_key: str | None = None,
) -> None
```

`ProviderAuth` is immutable per-request credential data. It only needs credentials for explicit-credential providers reachable from the request's `StageSelection` chain. Codex uses host auth files. Continuations must not store provider credentials.

### Outcomes and Continuations

#### `RuntimeOutcome`

```python
RuntimeOutcome(
    kind: str,
    output: str,
    result: EphemeralRunResult | SessionRunResult | None = None,
    service_name: str | None = None,
    account_label: str | None = None,
    reset_time: datetime | None = None,
    invocation_progress: InvocationProgress | None = None,
    continuation: Continuation | None = None,
    usage: ProviderUsage | None = None,
    invocation_records: tuple[InvocationRecord, ...] = (),
) -> None
```

Lifecycle entrypoints return `RuntimeOutcome` for both completed work and expected interruption outcomes.

Outcome kinds:

- `completed`: work completed and `result` is present.
- `usage_limited`: usage limit interrupted execution.
- `no_service_available`: configured candidates are temporarily unavailable.
- `cancelled`: caller- or user-initiated cancellation.
- `timed_out`: runtime timeout.
- `retryable_provider_failure`: provider failure classified as confidently retryable.

Expected interruption outcomes are normal lifecycle results. Credential failures, malformed inputs, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions remain errors.

Usage-limit outcomes expose provider and service facts such as selected service, account label, reset time, invocation progress, provider usage, and continuation state. Caller-defined usage-limit grouping is not part of the core runtime API.

#### `EphemeralRunResult`

```python
EphemeralRunResult(
    *,
    output: str,
    selected_service: str,
    selected_model: str,
    selected_effort: str,
    tool_policy: ToolPolicy,
    used_fallback: bool,
    usage: ProviderUsage | None = None,
) -> None
```

Ephemeral results report the selected provider facts and output text. They do not carry a continuation.

#### `SessionRunResult`

```python
SessionRunResult(
    *,
    output: str,
    runtime_metadata: SessionRuntimeMetadata,
    continuation: Continuation,
    usage: ProviderUsage | None = None,
) -> None
```

Session-backed results contain output text and a meaningful continuation. The continuation returned on a completed run is the latest resume token for the next resumed-session call.

#### `SessionRuntimeMetadata`

```python
SessionRuntimeMetadata(
    *,
    service_name: str,
    provider_session_id: str | None,
    run_kind: RunKind,
    selected_model: str,
    selected_effort: str,
    tool_policy: ToolPolicy,
    exact_transcript_match: bool,
) -> None
```

Display and policy metadata such as selected service, model, effort, and tool policy belongs in result metadata rather than in the continuation contract.

#### `Continuation`

```python
Continuation(serialized: str) -> None
```

`Continuation` is an opaque portable resume token. Callers may persist and pass it back to runtime entrypoints, but they must not inspect or depend on provider resume payloads as a stable schema.

The runtime may carry provider-owned resume data inside the continuation, including encoded provider state when needed. Provider credentials do not belong in continuations.

#### `InvocationRecord`

```python
InvocationRecord(
    *,
    run_kind: RunKind,
    service_name: str,
    provider_session_id: str | None,
    prompt: str,
    provider_output: bytes | None = None,
    usage: ProviderUsage | None = None,
) -> None
```

`InvocationRecord` is structured runtime output for callers that want to persist or display execution traces. The runtime returns records; callers own persistence, file layout, naming, redaction, retention, and cleanup.

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

#### `StageSelection`

```python
StageSelection(
    *,
    service: str,
    model: str,
    effort: str,
    fallback: StageSelection | None = None,
) -> None
```

One node in an ordered built-in provider candidate chain. The runtime validates every service/model/effort tuple.

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

## Adapter Surface

There is no supported consumer-defined provider adapter surface in the target runtime. Provider services, service registries, execution adapters, provider-session adapters, provider output DTOs, provider state capture/restore details, command builders, and provider event parsers are runtime-owned internals or maintainer-facing seams.

Public provider failure errors may expose `ProviderErrorObservation` for diagnostics. Consumers own storage, display, and redaction policy for those diagnostics.

## Transitional Concepts To Remove

The following pre-release concepts are transitional and should not remain in the ordinary consumer API:

- `InvocationRole`
- `UsageLimitScope`
- `RuntimeStateDir`
- `RuntimeLogsDir`
- `SessionNamespace`
- Provider state relative paths in continuation payloads
- Runtime-created durable invocation log files
- `ToolAccess`
- `ToolPolicyProfile`

The target public surface keeps ordinary consumers on `RuntimeClient`, request values, outcome values, built-in provider selection, `ProviderAuth`, `ProviderUsage`, `Continuation`, `InvocationRecord`, and `ToolPolicy`.
