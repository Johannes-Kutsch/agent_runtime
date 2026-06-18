# agent_runtime Public API

This is the documented `Runtime Public Surface` for `agent_runtime`. It is a stability target for ordinary runtime consumers and maintainers implementing issue #93, not an inventory of every importable symbol in the current pre-migration package.

The distribution name is `ruhken-agent-runtime`; the import package is `agent_runtime`. The package requires Python 3.11 or newer. The issue #93 target is that Claude, Codex, and OpenCode provider execution ships inside this distribution rather than in consuming projects or external adapter packages.

Pre-release runtime compatibility aliases are intentionally absent from the documented surface and from direct module attribute access. Current implementation seams may remain importable until the migration removes them, but importability does not make them consumer API.

## Consumer Surface

Ordinary consumers execute already-rendered prompts through a caller-owned `RuntimeClient`. They provide provider selection, credentials, tool access, worktree, and session lifecycle data as call arguments. They do not construct provider services, service registries, execution adapters, provider-session adapters, command builders, provider event parsers, or provider DTO streams.

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
- `InvocationRole`
- `ProviderAuth`
- `ProviderUsage`
- `RuntimeConfigurationError`
- `RuntimeOutcome`
- `RunKind`
- `StageSelection`
- `ToolAccess`
- `TransientAgentError`
- `UsageLimitError`
- `UsageLimitScope`

Behaviorful lifecycle entrypoints live in `agent_runtime.runtime`, not at the package root.

### Runtime Client

Import path: `agent_runtime.runtime`

```python
RuntimeClient() -> None

async run_ephemeral(request: EphemeralRunRequest) -> RuntimeOutcome
async run_new_session(request: NewSessionRunRequest) -> RuntimeOutcome
async run_resumed_session(request: ResumedSessionRunRequest) -> RuntimeOutcome
```

`RuntimeClient` holds in-process built-in provider availability and exhaustion state across calls. It does not own durable provider state, process-global auth setup, workflow defaults, prompt rendering, issue orchestration, managed worktrees, dependency installation, or application logging policy. It is safe to reuse across concurrent runtime requests and synchronizes provider availability updates internally.

### Built-In Providers

Provider selection uses `StageSelection` nodes. Every selected node must reference a built-in service and a model/effort value supported by that built-in integration.

| Service | Auth | Session state | Notes |
| --- | --- | --- | --- |
| `claude` | `ProviderAuth(claude_code_oauth_token=ClaudeCodeOAuthToken(...))` | Runtime-owned Claude Code state under `RuntimeStateDir` for session-backed runs | Invocation behavior should match pycastle's existing Claude service migration target. |
| `codex` | Host Codex auth files | Runtime-owned Codex state under `RuntimeStateDir` for session-backed runs | The runtime uses host auth files rather than API-key arguments. |
| `opencode` | `ProviderAuth(opencode_api_key=...)` | Runtime-owned OpenCode state under `RuntimeStateDir` for session-backed runs | Invocation behavior should match pycastle's existing OpenCode service migration target. |

The runtime validates built-in service, model, and effort values before provider execution. Invalid service/model/effort references are runtime configuration errors. Missing or invalid explicit credentials are credential failures and stop execution rather than triggering fallback.

Known migrated pycastle allowlists:

- Claude models: `haiku`, `sonnet`, `opus`; efforts: `low`, `medium`, `high`, `xhigh`, `max`.
- Codex models: `gpt-5.5`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.3-codex`, `gpt-5.3-codex-spark`, `gpt-5.2`; efforts: `low`, `medium`, `high`, `xhigh`.
- OpenCode models and command mappings come from pycastle's OpenCode service; effort is currently `medium`.

### Request Values

#### `EphemeralRunRequest`

```python
EphemeralRunRequest(
    *,
    prompt: str,
    worktree: Path,
    stage: StageSelection,
    role: InvocationRole,
    provider_auth: ProviderAuth,
    tool_access: ToolAccess,
    usage_limit_scope: UsageLimitScope | None = None,
    session_namespace: str = "",
    logs_dir: Path | None = None,
    token: CancellationToken | None = None,
) -> None
```

Ephemeral execution runs an already-rendered prompt without intentionally preparing provider-session continuity. It does not require `runtime_state_dir`.

#### `NewSessionRunRequest`

```python
NewSessionRunRequest(
    *,
    prompt: str,
    worktree: Path,
    runtime_state_dir: Path,
    stage: StageSelection,
    role: InvocationRole,
    provider_auth: ProviderAuth,
    tool_access: ToolAccess,
    usage_limit_scope: UsageLimitScope | None = None,
    session_namespace: str = "",
    logs_dir: Path | None = None,
    token: CancellationToken | None = None,
) -> None
```

New-session execution selects a built-in provider, prepares provider-native transcript continuity under `runtime_state_dir`, and returns a continuation for later resumed execution.

#### `ResumedSessionRunRequest`

```python
ResumedSessionRunRequest(
    *,
    prompt: str,
    worktree: Path,
    runtime_state_dir: Path,
    continuation: Continuation,
    role: InvocationRole,
    provider_auth: ProviderAuth,
    model: str | None = None,
    effort: str | None = None,
    usage_limit_scope: UsageLimitScope | None = None,
    logs_dir: Path | None = None,
    token: CancellationToken | None = None,
) -> None
```

Resumed-session execution continues an existing provider-session continuity chain. The continuation fixes service and tool access. Resumed execution does not perform fallback and rejects tool-access replacement; omitted model or effort values default from the continuation.

### Auth

#### `ProviderAuth`

```python
ProviderAuth(
    *,
    claude_code_oauth_token: ClaudeCodeOAuthToken | str | None = None,
    opencode_api_key: str | None = None,
) -> None
```

`ProviderAuth` is immutable per-request credential data. It only needs credentials for explicit-credential providers reachable from the request's `StageSelection` chain. Codex uses host auth files.

### Outcomes and Continuations

#### `RuntimeOutcome`

```python
RuntimeOutcome(
    kind: str,
    output: str,
    result: EphemeralRunResult | SessionRunResult | None = None,
    service_name: str | None = None,
    reset_time: datetime | None = None,
    usage_limit_scope: UsageLimitScope | None = None,
    invocation_progress: InvocationProgress | None = None,
    continuation: Continuation | None = None,
    usage: ProviderUsage | None = None,
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

#### `Continuation`

```python
Continuation(
    *,
    selected_service: str,
    selected_model: str,
    selected_effort: str,
    tool_access: ToolAccess,
    provider_resume_state: Any,
) -> None
```

Consumer-owned data needed to resume a provider-session continuity chain. `provider_resume_state` must be JSON-compatible. Continuations carry provider state identifiers relative to `RuntimeStateDir`, not absolute provider state paths.

### Runtime Value Objects

#### `InvocationRole`

Caller-defined invocation label. Values must be non-empty, path-safe labels without whitespace or path separators.

#### `UsageLimitScope`

Caller-defined grouping key for usage-limit continuation policy. It uses the same path-safe validation rules as `InvocationRole`.

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

#### `ToolAccess`

```python
ToolAccess.no_tools() -> ToolAccess
ToolAccess.workspace_backed(workspace: Path) -> ToolAccess
```

Closed runtime value for provider tool access. `ToolAccess` is the consumer-facing tool contract. Provider flag profiles, MCP config rendering, and command-line tool-policy mappings are internal built-in adapter policy.

Workspace-backed host subprocess execution runs the provider in the requested worktree. Tool-less execution still carries a worktree on the request but does not grant workspace-backed provider tool access. A provider-session continuity chain keeps the same `ToolAccess` for its lifetime.

## Adapter Surface

There is no supported consumer-defined provider adapter surface in the issue #93 target. The following concepts are runtime-owned internals or pre-migration implementation artifacts, not extension points for consuming projects:

- `ExecutionProvider`
- `ResumableExecutionProvider`
- `ServiceSelectionProvider`
- `SessionPlanningProvider`
- `ProviderStatePreparationAction`
- `ProviderSessionAdapter`
- `PromptRuntimeExecutionAdapter`
- `EphemeralRuntimeExecutionAdapter`
- `NewSessionRuntimeExecutionAdapter`
- `ResumedSessionRuntimeExecutionAdapter`
- `ServiceRegistry`
- `PromptRunRequest`
- `PromptRunSession`
- `RunSessionPlan`
- `TextOutputAdapter`
- `WorkExecutionAdapter`
- `WorkExecutionDependencies`
- `WorkFailureHandling`
- `WorkInvocationDependencies`
- `WorkInvocationPresentation`
- `WorkInvocationRequest`
- `WorkModelDisplayMetadata`
- `WorkOutputAdapter`
- `WorkPresentationDependencies`
- `WorkStatusDisplay`
- `WorkStatusRow`

Provider event DTOs are also internal built-in adapter details rather than consumer API:

- `AssistantTurn`
- `CredentialFailure`
- `HardError`
- `ModelActivity`
- `ParsedTurn`
- `PromptTokens`
- `Result`
- `TransientError`
- `UnsupportedTokens`
- `UsageLimit`
- `reduce_text_output_events`

Public provider failure errors may expose `ProviderErrorObservation` for diagnostics. Consumers own storage, display, and redaction policy for those diagnostics.

## Advanced Focused Seams

Advanced focused seams remain maintainer-facing while implementation slices migrate them behind the built-in execution substrate. They are not the README-first integration path and should not be used to create custom provider services.

Current pre-migration exported names that implementation slices must either internalize or intentionally re-document under the new model:

- `AgentCancelledError`
- `AgentCredentialFailureError`
- `AgentFailedError`
- `AgentRuntimeError`
- `AgentTimeoutError`
- `AuthSeedingRequirement`
- `CancellationToken`
- `Continuation`
- `EphemeralResultMetadata`
- `EphemeralRunRequest`
- `EphemeralRunResult`
- `EphemeralRuntime`
- `EphemeralRuntimeMetadata`
- `HardAgentError`
- `InvocationProgress`
- `InvocationRole`
- `LocalAuthSeedAction`
- `LogicalAgentInvocationLog`
- `NewSessionRunRequest`
- `NewSessionRuntime`
- `NoServiceAvailableError`
- `PrepareSessionAdapter`
- `PreparedProviderRunSession`
- `PreparedRunSessionState`
- `PreparedSession`
- `ProviderAccountExhaustionHandler`
- `ProviderErrorObservation`
- `ProviderSessionDecision`
- `ProviderSessionPlanRequest`
- `ProviderSessionPlanningFacts`
- `ProviderSessionPlanningRequest`
- `ProviderSessionSelection`
- `ProviderSessionState`
- `ProviderSessionStateRequest`
- `RecoveredSessionIdPersistence`
- `ResumabilityProvider`
- `ResumableSessionPlan`
- `ResumableSessionPlanRequest`
- `ResumedSessionRunRequest`
- `ResumedSessionRuntime`
- `RetryableProviderFailureError`
- `RunKind`
- `RuntimeConfigurationError`
- `RuntimeOutcome`
- `SessionRunResult`
- `SessionRuntimeMetadata`
- `SessionStore`
- `SetupFailureTranslator`
- `StageSelection`
- `StatusDisplayFactory`
- `StatusRowFactory`
- `ToolAccess`
- `ToolPolicy`
- `ToolPolicyProfile`
- `TransientAgentError`
- `UsageLimitError`
- `UsageLimitScope`
- `WorkInvocationLog`
- `WorkResultT`
- `WorktreeMount`
- `AgentInvocationLog`
- `is_exact_resumable_service_session`
- `load_provider_state_session_id`
- `load_state_dir_provider_session_id`
- `normalize_state_dir_relpath`
- `plan_provider_session`
- `plan_resumable_session`
- `provider_state_relpath`
- `provider_state_session_id_path`
- `select_resumable_provider_session_id`

The target public surface keeps ordinary consumers on `RuntimeClient`, request values, outcome values, built-in provider selection, `ProviderAuth`, `ProviderUsage`, `Continuation`, and `ToolAccess`.
