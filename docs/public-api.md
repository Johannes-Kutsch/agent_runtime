# agent_runtime Public API

This is the documented `Runtime Public Surface` for `agent_runtime`. It is a stability promise for ordinary runtime consumers, adapter authors, and advanced focused seams. It is not an inventory of every importable symbol in the package.

The distribution name is `ruhken-agent-runtime`; the import package is `agent_runtime`. The package requires Python 3.11 or newer and has no install-time provider dependencies. Provider-specific adapters live outside this package.

Pre-release runtime compatibility aliases are intentionally absent from the documented surface and from direct module attribute access. Canonical lifecycle entrypoints and focused seams are the public contract.

## Consumer Surface

Ordinary consumers should use the lifecycle entrypoints in `agent_runtime.runtime` with package-root vocabulary.

### Package Root

Import path: `agent_runtime`

The package root exposes stable shared vocabulary and common errors:

- `AgentCredentialFailureError`
- `AgentFailedError`
- `AgentRuntimeError`
- `AgentTimeoutError`
- `Continuation`
- `ExecutionProvider`
- `HardAgentError`
- `InvocationProgress`
- `InvocationRole`
- `ProviderSessionAdapter`
- `RunKind`
- `RuntimeConfigurationError`
- `RuntimeOutcome`
- `StageSelection`
- `ToolAccess`
- `ToolPolicy`
- `ToolPolicyProfile`
- `TransientAgentError`
- `UsageLimitError`
- `UsageLimitScope`

Behaviorful lifecycle entrypoints live in `agent_runtime.runtime`, not at the package root.

### Runtime Lifecycle Entrypoints

Import path: `agent_runtime.runtime`

#### `EphemeralRuntime`

```python
EphemeralRuntime(
    *,
    execution_adapter: EphemeralRuntimeExecutionAdapter,
    service_registry: ServiceRegistry | dict[str, Any] | None = None,
) -> None

async run_ephemeral(request: EphemeralRunRequest) -> RuntimeOutcome
```

Runs an already-rendered prompt without preparing provider-session continuity. `execution_adapter` resolves services and builds work dependencies. `service_registry` may be a `ServiceRegistry`, a service mapping, or omitted.

#### `EphemeralRunRequest`

```python
EphemeralRunRequest(
    prompt: str,
    worktree: Path | WorktreeMount,
    stage: StageSelection | None = None,
    role: InvocationRole | None = None,
    usage_limit_scope: UsageLimitScope | None = None,
    tool_policy: ToolPolicy | ToolPolicyProfile | object = MISSING,
    tool_access: ToolAccess | object = MISSING,
    session_namespace: str = "",
    token: CancellationToken | None = None,
    *,
    override: StageSelection | None = None,
) -> None
```

Parameters:

- `prompt`: already-rendered prompt text.
- `worktree`: host worktree path or `WorktreeMount`.
- `stage`: service, model, effort, and fallback chain. `override` is a compatibility spelling for `stage`; do not pass conflicting values.
- `role`: caller-supplied invocation label.
- `usage_limit_scope`: optional caller-defined usage-limit grouping. Defaults to the invocation role where runtime policy needs a scope.
- `tool_access`: explicit tool access. Use `ToolAccess.no_tools()` for tool-less execution or `ToolAccess.workspace_backed(worktree)` for workspace-backed execution.
- `tool_policy`: compatibility shortcut that builds workspace-backed tool access. Prefer `tool_access`.
- `session_namespace`: optional path-safe namespace; the empty namespace is the default.
- `token`: cooperative cancellation token.

Invariants:

- `stage` and `role` are required.
- Tool access is required.
- Workspace-backed `ToolAccess` must match `worktree`.
- Non-empty `session_namespace` values must be path-safe labels.

Properties:

- `mount_path -> Path`: host worktree path.
- `override -> StageSelection`: compatibility property for `stage`.
- `tool_policy -> ToolPolicy | ToolPolicyProfile`: policy carried by `tool_access`.

#### `EphemeralRunResult`

```python
EphemeralRunResult(
    output: str,
    selected_service: str,
    selected_model: str,
    selected_effort: str,
    tool_access: ToolAccess,
    used_fallback: bool,
    metadata: EphemeralResultMetadata,
) -> None
```

Completed ephemeral result data. `raw_output` is an alias for `output`. `selected_service_path` and `runtime_metadata` are derived from `metadata`.

#### `EphemeralResultMetadata`

```python
EphemeralResultMetadata(
    selected_service_path: tuple[str, ...],
    runtime: EphemeralRuntimeMetadata,
) -> None
```

Groups service-selection diagnostics and runtime metadata.

#### `EphemeralRuntimeMetadata`

```python
EphemeralRuntimeMetadata(
    run_kind: RunKind,
    session_namespace: str,
) -> None
```

Runtime metadata for completed ephemeral execution.

#### `NewSessionRuntime`

```python
NewSessionRuntime(
    *,
    execution_adapter: NewSessionRuntimeExecutionAdapter,
    service_registry: ServiceRegistry | dict[str, Any] | None = None,
) -> None

async run_new_session(request: NewSessionRunRequest) -> RuntimeOutcome
```

Runs a prompt while preparing provider-session continuity and returning a continuation for later process calls.

#### `NewSessionRunRequest`

```python
NewSessionRunRequest(
    prompt: str,
    worktree: Path | WorktreeMount,
    stage: StageSelection | None = None,
    role: InvocationRole | None = None,
    session_store: Any | None = None,
    provider_session_adapter: ProviderSessionAdapter | None = None,
    usage_limit_scope: UsageLimitScope | None = None,
    tool_policy: ToolPolicy | ToolPolicyProfile | object = MISSING,
    tool_access: ToolAccess | object = MISSING,
    session_namespace: str = "",
    name: str = "Runtime Agent",
    status_display: Any = None,
    work_body: str = "",
    token: CancellationToken | None = None,
    *,
    override: StageSelection | None = None,
) -> None
```

Parameters:

- `prompt`, `worktree`, `stage`, `role`, `usage_limit_scope`, `tool_access`, `tool_policy`, `session_namespace`, `token`, and `override`: same meanings as `EphemeralRunRequest`.
- `session_store`: consumer-owned persistence seam for provider-session identity.
- `provider_session_adapter`: provider-specific session planning and recording seam.
- `name`, `status_display`, `work_body`: presentation inputs for advanced adapters.

Invariants:

- `stage`, `role`, `session_store`, `provider_session_adapter`, and explicit tool access are required.
- Workspace-backed `ToolAccess` must match `worktree`.
- Returned continuations are consumer-owned; the runtime does not persist them.

Properties:

- `mount_path -> Path`
- `override -> StageSelection`
- `tool_policy -> ToolPolicy | ToolPolicyProfile`

#### `ResumedSessionRuntime`

```python
ResumedSessionRuntime(
    *,
    execution_adapter: ResumedSessionRuntimeExecutionAdapter,
) -> None

async run_resumed_session(request: ResumedSessionRunRequest) -> RuntimeOutcome
```

Runs a prompt against an existing continuation without service fallback or tool-policy replacement.

#### `ResumedSessionRunRequest`

```python
ResumedSessionRunRequest(
    prompt: str,
    worktree: WorktreeMount,
    model: str | None = None,
    effort: str | None = None,
    session_plan: ResumableSessionPlan | None = None,
    continuation: Continuation | None = None,
    role: InvocationRole | None = None,
    session_namespace: str = "",
    usage_limit_scope: UsageLimitScope | None = None,
    tool_policy: ToolPolicy | object = MISSING,
    tool_access: ToolAccess | object = MISSING,
    name: str = "Runtime Agent",
    status_display: Any = None,
    work_body: str = "",
    token: CancellationToken | None = None,
) -> None
```

Parameters:

- `prompt`: already-rendered prompt text for the resumed call.
- `worktree`: mounted host worktree.
- `continuation`: latest continuation from a new-session or resumed-session result.
- `role`: caller-supplied invocation label required when resuming from a continuation.
- `model`, `effort`: optional overrides; omitted values default from the continuation.
- `session_namespace`, `usage_limit_scope`, `name`, `status_display`, `work_body`, `token`: runtime metadata and advanced presentation inputs.
- `session_plan`, `tool_policy`, `tool_access`: lower-level construction path. Ordinary resumed-session consumers should pass `continuation` and not pass these.

Invariants:

- `continuation` and `session_plan` are mutually exclusive.
- Continuation-based construction requires `role`.
- Continuation-based construction derives fixed tool access from `continuation` and rejects `tool_access` or `tool_policy` overrides.
- Workspace-backed continuation tool access must match `worktree.host_path`.

Properties:

- `mount_path -> Path`
- `tool_policy -> ToolPolicy | ToolPolicyProfile`

#### `SessionRunResult`

```python
SessionRunResult(
    output: str,
    runtime_metadata: SessionRuntimeMetadata,
    continuation: Continuation | None = None,
) -> None
```

Completed result for session-backed execution. Store `continuation` if the consuming project wants to resume later.

#### `SessionRuntimeMetadata`

```python
SessionRuntimeMetadata(
    service_name: str,
    provider_session_id: str | None,
    run_kind: RunKind,
    session_namespace: str,
    exact_transcript_match: bool,
) -> None
```

Session-backed runtime metadata for completed new-session or resumed-session execution.

#### Runtime Execution Adapter Aliases

```python
EphemeralRuntimeExecutionAdapter = PromptRuntimeExecutionAdapter
NewSessionRuntimeExecutionAdapter = PromptRuntimeExecutionAdapter
ResumedSessionRuntimeExecutionAdapter = PromptRuntimeExecutionAdapter
```

These aliases name the adapter contract expected by each lifecycle runtime.

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
) -> None
```

Lifecycle entrypoints return `RuntimeOutcome` for both completed work and expected interruption outcomes.

Factory constructors:

```python
RuntimeOutcome.completed(*, output: str, result: EphemeralRunResult | SessionRunResult) -> RuntimeOutcome
RuntimeOutcome.usage_limited(*, output: str, service_name: str | None, reset_time: datetime | None, usage_limit_scope: UsageLimitScope | None, invocation_progress: InvocationProgress, continuation: Continuation | None = None) -> RuntimeOutcome
RuntimeOutcome.no_service_available(*, output: str, reset_time: datetime | None, usage_limit_scope: UsageLimitScope | None = None, invocation_progress: InvocationProgress, continuation: Continuation | None = None) -> RuntimeOutcome
RuntimeOutcome.cancelled(*, output: str, invocation_progress: InvocationProgress, continuation: Continuation | None = None) -> RuntimeOutcome
RuntimeOutcome.timed_out(*, output: str, invocation_progress: InvocationProgress, continuation: Continuation | None = None) -> RuntimeOutcome
RuntimeOutcome.retryable_provider_failure(*, output: str, service_name: str, invocation_progress: InvocationProgress, continuation: Continuation | None = None) -> RuntimeOutcome
```

Outcome kinds:

- `completed`: work completed and `result` is present.
- `usage_limited`: usage limit interrupted execution.
- `no_service_available`: configured candidates are temporarily unavailable.
- `cancelled`: caller- or user-initiated cancellation.
- `timed_out`: runtime timeout.
- `retryable_provider_failure`: provider failure classified as confidently retryable.

Convenience properties:

- `runtime_metadata`
- `selected_service_path`
- `selected_service`
- `selected_model`
- `selected_effort`
- `used_fallback`
- `tool_access`
- `raw_output`

These properties raise `AttributeError` when the outcome does not carry the relevant completed result shape.

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

Consumer-owned data needed to resume a provider-session continuity chain. `provider_resume_state` must be JSON-compatible and is exposed through the `provider_resume_state` property as decoded data. From the consumer perspective, continuations are portable but semantically immutable.

### Runtime Value Objects

#### `InvocationRole`

```python
InvocationRole(value: str) -> None
```

Caller-defined invocation label. `value` must be non-empty, not whitespace-only, contain no whitespace, contain no path separators, and not be `.` or `..`.

#### `UsageLimitScope`

```python
UsageLimitScope(value: str) -> None
```

Caller-defined grouping key for usage-limit continuation policy. It uses the same path-safe validation rules as `InvocationRole`.

#### `StageSelection`

```python
StageSelection(
    model: str = "",
    effort: str = "",
    service: str = "",
    fallback: StageSelection | None = None,
) -> None
```

One node in an ordered stage chain. Every node must provide non-empty `service`, `model`, and `effort`. `service` must be path-safe. `fallback` links to the next candidate.

#### `ToolPolicyProfile`

```python
ToolPolicyProfile(
    allowed_tools: tuple[str, ...] | None = None,
    disallowed_tools: tuple[str, ...] = (),
    strict_mcp_config: bool = True,
) -> None
```

Provider-neutral tool policy profile. Provider adapters translate it to provider-specific command flags or restrictions.

#### `ToolPolicy`

```python
ToolPolicy.RESTRICTED
ToolPolicy.PARTIAL
ToolPolicy.FULL

ToolPolicy.profile -> ToolPolicyProfile
```

Closed coarse tool-policy enum. `RESTRICTED` allows read/glob-style access, `PARTIAL` forbids editing/writing tools, and `FULL` has no runtime-owned tool restrictions.

#### `ToolAccess`

```python
ToolAccess(
    *,
    kind: str,
    workspace: Path | None,
    tool_policy: ToolPolicy | ToolPolicyProfile,
) -> None

ToolAccess.no_tools() -> ToolAccess
ToolAccess.workspace_backed(workspace: Path, *, tool_policy: ToolPolicy | ToolPolicyProfile = ToolPolicy.FULL) -> ToolAccess
tool_policy -> ToolPolicy | ToolPolicyProfile
require_workspace(workspace: Path | None, *, context: str) -> None
```

Closed runtime value for provider tool access. `kind` is `none` or `workspace_backed`. `no_tools()` forbids provider tool access and carries no workspace. `workspace_backed()` requires a workspace path and validates that tool access matches the request worktree.

#### `WorktreeMount`

```python
WorktreeMount(host_path: Path) -> None
```

Host worktree mount value used by resumed-session and advanced execution APIs.

#### `InvocationProgress`

```python
InvocationProgress.NOT_STARTED
InvocationProgress.STARTED
```

Two-state runtime outcome metadata. Expected interruptions report whether model activity had started.

#### `RunKind`

```python
RunKind.FRESH
RunKind.RESUME
```

Closed runtime session lifecycle enum.

## Adapter Surface

Adapter authors implement these contracts to connect the runtime to provider or application infrastructure.

### Provider Contracts

Import path: `agent_runtime.contracts`

#### Provider Event DTOs

```python
AssistantTurn(text: str) -> None
PromptTokens(count: int) -> None
UnsupportedTokens(count: int, source: str) -> None
Result(text: str) -> None
ModelActivity() -> None
UsageLimit(reset_time: datetime | None, raw_message: str | None = None, is_permanent: bool = False) -> None
TransientError(status_code: int | None, raw_message: str, classification: str | None = None, observations: tuple[ProviderErrorObservation, ...] = ()) -> None
HardError(status_code: int, raw_message: str, classification: str | None = None, observations: tuple[ProviderErrorObservation, ...] = ()) -> None
CredentialFailure(raw_message: str, service_name: str, source_observations: tuple[ProviderErrorObservation, ...], status_code: int | None = None, classification: str | None = None) -> None
```

Provider adapters emit these DTOs so runtime-owned output reduction can classify text, token counts, usage limits, retryable failures, hard failures, and credential failures.

`ParsedTurn` is the union of these provider event DTOs.

#### Provider Protocols

```python
class ProviderStatePreparationAction(Protocol):
    apply(self) -> None

class ServiceSelectionProvider(Protocol):
    is_available(self, now: datetime | None = None) -> bool
    next_wake_time(self) -> datetime
    mark_exhausted(self, reset_time: datetime | None) -> None

class ResumabilityProvider(Protocol):
    is_resumable(self, state_dir: Path) -> bool

class ExecutionProvider(Protocol):
    name -> str
    build_command(self, role: InvocationRole, model: str, effort: str, run_kind: RunKind, session_uuid: str | None, *, tool_policy: ToolPolicy | ToolPolicyProfile | Any | None = None) -> str
    build_env(self, state_dir_container_path: str | None = None, token: str | None = None) -> dict[str, str]
    run(self, lines: Iterable[str], on_provider_session_id: Callable[[str], None] | None = None) -> Iterator[ParsedTurn]
    mark_exhausted(self, reset_time: datetime | None) -> None

class ResumableExecutionProvider(ResumabilityProvider, ExecutionProvider, Protocol):
    pass

class SessionPlanningProvider(ResumabilityProvider, Protocol):
    name -> str
```

`ExecutionProvider` is the primary provider execution protocol. `ServiceSelectionProvider` owns availability policy for `ServiceRegistry`. `ResumabilityProvider` and `SessionPlanningProvider` support provider-session planning.

### Runtime Execution Contracts

Import path: `agent_runtime.execution_contracts`

These are advanced adapter execution contracts. They are public because adapter authors need them, but ordinary consumers should prefer lifecycle runtimes.

#### Request and Session Values

```python
PromptRunSession(namespace: str = "", plan: Any = None) -> None
PromptRunRequest(prompt: str, worktree: WorktreeMount, stage: StageSelection | None = None, role: InvocationRole | None = None, usage_limit_scope: UsageLimitScope | None = None, tool_policy: ToolPolicy | object = MISSING, tool_access: ToolAccess | object = MISSING, name: str = "Runtime Agent", status_display: Any = None, work_body: str = "", token: CancellationToken | None = None, session: PromptRunSession | None = None, *, override: StageSelection | None = None) -> None
RunSessionPlan(mount_path: Path, role: InvocationRole, session_namespace: str, service: ExecutionProvider, container_workspace: str, usage_limit_scope: UsageLimitScope | None = None, run_kind: RunKind = RunKind.FRESH, provider_session_id: str | None = None, provider_resume_state: Any = None, provider_state_dir_container_path: str | None = None, exact_transcript_match: bool = False, run_session_plan: Any = None) -> None
WorkInvocationPresentation(name: str = "Runtime Agent", status_display: Any = None, work_body: str = "", color_key: int | None = None) -> None
WorkInvocationRequest(run_session: RunSessionPlan, model: str, effort: str, output_adapter: WorkOutputAdapter[WorkResultT], dependencies: WorkInvocationDependencies, presentation: WorkInvocationPresentation = WorkInvocationPresentation(), token: CancellationToken | None = None, allow_non_typed_resume_retry: bool = False) -> None
```

`PromptRunRequest` validates `stage`, `role`, tool access, and session namespace. Its properties are `mount_path`, `tool_policy`, `override`, `session_namespace`, and `run_session_plan`.

`WorkInvocationRequest` exposes `name`, `status_display`, `work_body`, `color_key`, `mount_path`, `role`, `service`, and `session_namespace` as convenience properties over nested values.

#### Adapter Protocols

```python
class PromptRuntimeExecutionAdapter(Protocol):
    resolve_service(self, service_name: str = "") -> ExecutionProvider
    build_work_dependencies(self, *, name: str, model: str, effort: str, service: ExecutionProvider) -> WorkInvocationDependencies

class PreparedProviderRunSession(Protocol):
    run_kind: RunKind
    provider_session_id: str | None
    record_provider_session_id(self, provider_session_id: str) -> None
    record_successful_run(self) -> None

class PreparedRunSessionState(Protocol):
    provider_state_dir_container_path: str | None
    prepare_for_run(self) -> None
    initial_provider_run_session(self) -> PreparedProviderRunSession
    resumable_provider_run_session(self) -> PreparedProviderRunSession
    protocol_reprompt_provider_run_session(self) -> PreparedProviderRunSession | None

class WorkStatusDisplay(Protocol):
    register(self, caller: str, kind: str, startup_message: str = "started", work_body: str = "", initial_phase: str = "Setup", color_key: int | None = None, model_display: WorkModelDisplayMetadata | None = None) -> None
    update_phase(self, name: str, phase: str) -> None
    reset_idle_timer(self, name: str) -> None
    update_tokens(self, name: str, current_tokens: int) -> None
    remove(self, caller: str, shutdown_message: str = "finished", shutdown_style: str = "success") -> None
    print(self, caller: str, message: object, style: str | None = None) -> None

class WorkStatusRow(Protocol):
    close(self, shutdown_message: str = "finished", *, shutdown_style: str = "success") -> None

class WorkExecutionAdapter(Protocol):
    async setup(self, git_name: str, git_email: str, work_body: str = "") -> None
    async prompt_only(self, prompt: str, *, role: InvocationRole = InvocationRole("implementer"), run_kind: RunKind = RunKind.FRESH, session_uuid: str | None = None, on_provider_session_id: Callable[[str], None] | None = None) -> Any
    async work(self, role: InvocationRole, prompt: str, *, run_kind: RunKind = RunKind.FRESH, session_uuid: str | None = None, on_provider_session_id: Callable[[str], None] | None = None) -> Any
    async work_text(self, prompt: str, *, role: InvocationRole = InvocationRole("implementer"), tool_policy: ToolPolicy | ToolPolicyProfile, run_kind: RunKind = RunKind.FRESH, session_uuid: str | None = None, on_provider_session_id: Callable[[str], None] | None = None) -> str

class WorkOutputAdapter(Protocol[WorkResultT]):
    async build_prompt(self, *, run_kind: RunKind, container_exec: Callable[[str], Awaitable[str]]) -> str
    async invoke(self, *, runner: WorkExecutionAdapter, role: InvocationRole, prompt: str, run_kind: RunKind, session_uuid: str | None, on_provider_session_id: Callable[[str], None]) -> WorkResultT
    is_successful_result(self, result: WorkResultT) -> bool
    protocol_reprompt_message(self) -> str | None
    protocol_error_result(self) -> WorkResultT | None
    protocol_error_types(self) -> tuple[type[BaseException], ...]
    non_typed_failure_result(self) -> WorkResultT | None
    finalize_result(self, result: WorkResultT, *, role: InvocationRole, mount_path: Path, session_namespace: str, service_name: str) -> WorkResultT
```

#### Dependency Values and Aliases

```python
WorkModelDisplayMetadata(service: str, model: str, effort: str) -> None
WorkExecutionDependencies(container_workspace: str, prepare_session: PrepareSessionAdapter, build_session: Callable[[Path, ExecutionProvider, str | None], Any], build_runner: Callable[[Any, Any], WorkExecutionAdapter], get_git_identity: Callable[[], tuple[str, str]]) -> None
WorkFailureHandling(timeout_retries: int, translate_setup_failure: SetupFailureTranslator | None = None, handle_provider_account_exhaustion: ProviderAccountExhaustionHandler = default, transient_status_message: Callable[[Any], str] | None = None) -> None
WorkPresentationDependencies(status_display_factory: StatusDisplayFactory = default, status_row_factory: StatusRowFactory = default, build_model_display_metadata: Callable[[str, str, str], Any | None] | None = None) -> None
WorkInvocationDependencies(execution: WorkExecutionDependencies, failure_handling: WorkFailureHandling, presentation: WorkPresentationDependencies = WorkPresentationDependencies()) -> None
CancellationToken() -> None
```

Aliases:

- `WorkResultT`
- `PreparedSession = PreparedRunSessionState`
- `PrepareSessionAdapter = Callable[[RunSessionPlan], PreparedRunSessionState]`
- `StatusRowFactory = Callable[..., AbstractAsyncContextManager[Any]]`
- `SetupFailureTranslator = Callable[[InvocationRole, BaseException], BaseException | None]`
- `ProviderAccountExhaustionHandler = Callable[[ExecutionProvider, Any], None]`
- `StatusDisplayFactory = Callable[[], WorkStatusDisplay]`

`CancellationToken` has `is_cancelled -> bool` and `cancel() -> None`.

#### `TextOutputAdapter`

```python
TextOutputAdapter(
    prompt: str,
    tool_policy: ToolPolicy | ToolPolicyProfile | object = MISSING,
    tool_access: ToolAccess | object = MISSING,
    workspace: Path | None = None,
) -> None

async build_prompt(*, run_kind: RunKind, container_exec: Callable[[str], Awaitable[str]]) -> str
async invoke(*, runner: WorkExecutionAdapter, role: InvocationRole, prompt: str, run_kind: RunKind, session_uuid: str | None, on_provider_session_id: Callable[[str], None]) -> str
is_successful_result(result: str) -> bool
protocol_reprompt_message() -> str | None
protocol_error_result() -> str | None
non_typed_failure_result() -> str | None
protocol_error_types() -> tuple[type[BaseException], ...]
finalize_result(result: str, *, role: InvocationRole, mount_path: Path, session_namespace: str, service_name: str) -> str
```

Text output adapter for already-rendered prompts. It requires explicit tool policy or `ToolAccess`.

### Provider Session Adapter

Import path: `agent_runtime.provider_session_adapter`

```python
ProviderSessionPlanningRequest(worktree: Path, role: InvocationRole, namespace: str) -> None
ProviderSessionPlanningFacts(state_dir_relpath: str | None, provider_state_dir: Path | None, has_resumable_provider_state: bool) -> None

class ProviderSessionAdapter(Protocol):
    service_name -> str
    provider_session_planning_facts(self, request: ProviderSessionPlanningRequest) -> ProviderSessionPlanningFacts
    provider_session_state(self, request: ProviderSessionStateRequest) -> ProviderSessionState
    prepare_local_provider_run_state(self, provider_state_dir: Path | None, auth_seed_action: ProviderStatePreparationAction | None = None) -> None
    record_provider_session_id(self, *, session_store: SessionStore, provider_session_id: str, service_state_dir: Path | None = None) -> None
```

Provider adapters own provider-specific session policy and state preparation. The runtime owns the continuation boundary.

### Provider Output and Errors

Import paths: `agent_runtime.provider_output`, `agent_runtime.provider_errors`

```python
reduce_text_output_events(
    events: Iterable[ParsedTurn],
    on_turn: Callable[[str], None],
    on_tokens: Callable[[int], None] | None = None,
    *,
    provider: str,
) -> str

ProviderErrorObservation(
    service_name: str,
    raw_provider_text: str,
    source_stream: str,
    status_code: int | None = None,
    provider_code: str | None = None,
    error_name: str | None = None,
) -> None
```

`reduce_text_output_events` reduces provider events to text and maps provider failure events to runtime-owned errors. `ProviderErrorObservation` preserves raw diagnostic observations for caller-owned display, storage, and redaction policy.

## Advanced Focused Seams

These APIs are public for consumers or adapter authors assembling service selection, session planning, provider output, or log lifecycle behavior directly. They are not the README-first integration path.

### Service Registry

Import path: `agent_runtime.service_registry`

```python
ServiceRegistry(services: Mapping[str, ServiceSelectionProvider]) -> None

services -> dict[str, ServiceSelectionProvider]
has_configured_candidate(override: StageSelection) -> bool
resolve(override: StageSelection, now: datetime) -> StageSelection
has_available(now: datetime) -> bool
has_available_for(override: StageSelection, now: datetime) -> bool
next_wake_time(now: datetime) -> datetime | None
next_wake_time_for(override: StageSelection, now: datetime) -> datetime | None
mark_exhausted(service_name: str, *, reset_time: datetime | None) -> None
__getitem__(key: str) -> ServiceSelectionProvider | None
```

Runtime-owned resolver for configured services and stage chains. Service names must be path-safe runtime identity labels.

### Session Store and Provider Session Helpers

Import path: `agent_runtime.session`

```python
class SessionStore(Protocol):
    session_uuid(self) -> str
    service_session_id(self, service_name: str) -> str | None
    save_service_session_id(self, service_name: str, session_id: str) -> None
    service_session_metadata(self, service_name: str) -> dict[str, str] | None
    exact_transcript_service_name(self) -> str | None

ProviderSessionSelection(provider_session_id: str | None, persist_provider_session_id: bool = False) -> None
ProviderSessionStateRequest(session_store: SessionStore, provider_state_dir: Path | None, has_resumable_provider_state: bool, state_dir_relpath: str | None = None, require_exact_transcript_match: bool = False) -> None
ProviderSessionState(run_kind: RunKind, provider_session_id: str | None, state_dir_relpath: str | None = None, state_dir_path: Path | None = None, exact_transcript_match: bool = False, persist_provider_session_id: bool = False, auth_seeding_requirement: AuthSeedingRequirement | None = None, auth_seed_action: LocalAuthSeedAction | None = None, use_service_state_dir_for_container: bool = False) -> None
```

Functions:

```python
provider_state_relpath(role: InvocationRole, provider_name: str, namespace: str = "", *, session_root: str = "") -> str
normalize_state_dir_relpath(role: InvocationRole, namespace: str, service_name: str, state_dir_relpath: str | None, *, session_root: str | None = None) -> str | None
provider_state_session_id_path(state_dir: Path, service_name: str, *, session_id_filename: str = "thread_id") -> Path
load_provider_state_session_id(path: Path) -> str | None
load_state_dir_provider_session_id(state_dir: Path | None, service_name: str, *, session_id_filename: str = "thread_id") -> str | None
select_resumable_provider_session_id(session_store: SessionStore, service_name: str, *, provider_state_dir: Path | None, has_resumable_provider_state: bool, recover_provider_session_id: Callable[[Path | None], str | None] | None = None) -> ProviderSessionSelection
is_exact_resumable_service_session(session_store: SessionStore, service_name: str, *, provider_session_id: str | None, provider_state_dir: Path | None, exact_provider_session_matcher: Callable[[str | None, Path | None], bool] | None = None) -> bool
```

These helpers implement provider-session path, recovery, and exact-resume policy.

### Session Planning

Import path: `agent_runtime.session_planning`

```python
AuthSeedingRequirement.REQUIRED
AuthSeedingRequirement.NOT_REQUIRED

RecoveredSessionIdPersistence.PERSIST
RecoveredSessionIdPersistence.SKIP

LocalAuthSeedAction(source: Path, destination: Path, missing_source_message: str | None = None, missing_source_service_name: str | None = None, missing_source_status_code: int | None = None, missing_source_classification: str | None = None, missing_source_observations: tuple[ProviderErrorObservation, ...] = ()) -> None
require_source() -> Path
apply() -> None

ProviderSessionDecision(run_kind: RunKind, provider_session_id: str | None, state_dir_relpath: str | None, state_dir_path: Path | None, recovered_session_id_persistence: RecoveredSessionIdPersistence, service_state_dir: Path | None = None, exact_transcript_match: bool = False, auth_seeding_requirement: AuthSeedingRequirement = AuthSeedingRequirement.NOT_REQUIRED, auth_seed_action: LocalAuthSeedAction | None = None, use_service_state_dir_for_container: bool = False) -> None
container_state_dir() -> Path | None
container_state_dir_path(*, worktree: Path, container_workspace: str) -> str | None

ProviderSessionPlanRequest(worktree: Path, role: InvocationRole, namespace: str, resumability_service: ResumabilityProvider, session_store: SessionStore, provider_session_adapter: ProviderSessionAdapter) -> None
ResumableSessionPlanRequest(worktree: Path, role: InvocationRole, namespace: str, service: ExecutionProvider, session_store: SessionStore, provider_session_adapter: ProviderSessionAdapter, resumability_service: ResumabilityProvider | None = None, usage_limit_scope: UsageLimitScope | None = None) -> None
ResumableSessionPlan(role: InvocationRole, worktree: Path, namespace: str, service: ExecutionProvider, run_kind: RunKind, provider_state_dir: Path | None, provider_session_id: str | None, auth_seeding_requirement: AuthSeedingRequirement, auth_seed_action: LocalAuthSeedAction | None = None, exact_transcript_match: bool = False, usage_limit_scope: UsageLimitScope | None = None) -> None
```

Functions:

```python
plan_provider_session(request: ProviderSessionPlanRequest) -> ProviderSessionDecision
plan_resumable_session(request: ResumableSessionPlanRequest) -> ResumableSessionPlan
```

Session planning composes runtime-owned path and recovery policy with provider-owned session decisions.

### Agent Invocation Logging

Import path: `agent_runtime.agent_log`

```python
AgentInvocationLog(*, now_local: Callable[[], datetime] | None = None) -> None
reserve(*, log_name: str, logs_dir: Path) -> Path
start_logical_session(*, log_name: str, logs_dir: Path) -> LogicalAgentInvocationLog
open_work_invocation(*, log_path: Path, role: InvocationRole, run_kind: RunKind, session_uuid: str | None, prompt: str, usage_limit_scope: UsageLimitScope | None = None) -> Iterator[WorkInvocationLog]
append_work_invocation(*, log_path: Path, role: InvocationRole, run_kind: RunKind, session_uuid: str | None, prompt: str, provider_bytes: bytes, usage_limit_scope: UsageLimitScope | None = None) -> None

LogicalAgentInvocationLog(owner: AgentInvocationLog, *, log_path: Path) -> None
open_work_invocation(*, role: InvocationRole, run_kind: RunKind, session_uuid: str | None, prompt: str, usage_limit_scope: UsageLimitScope | None = None) -> Iterator[WorkInvocationLog]
append_work_invocation(*, role: InvocationRole, run_kind: RunKind, session_uuid: str | None, prompt: str, provider_bytes: bytes, usage_limit_scope: UsageLimitScope | None = None) -> None
record_provider_session_id(provider_session_id: str | None) -> None

WorkInvocationLog(log: BinaryIO, *, log_path: Path, header_start: int, header_record: dict[str, object]) -> None
append_provider_chunk(provider_bytes: bytes) -> None
record_provider_session_id(provider_session_id: str | None) -> None
```

The runtime owns log record shape and append/update lifecycle. Consuming applications own log location, presentation, retention, and redaction policy.

### Errors

Import path: `agent_runtime.errors`

Expected interruption errors are normally translated into `RuntimeOutcome` by lifecycle entrypoints. Advanced adapter seams may still raise or inspect them.

```python
AgentRuntimeError(message: str = "") -> None
RuntimeConfigurationError(message: str = "") -> None
AgentCancelledError(*, invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED, continuation: Any | None = None) -> None
AgentTimeoutError(message: str = "", invocation_role: str = "", worktree_path: Path | None = None, invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED, continuation: Any | None = None) -> None
NoServiceAvailableError(*, reset_time: datetime | None = None, usage_limit_scope: UsageLimitScope | None = None, invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED, continuation: Any | None = None) -> None
UsageLimitError(reset_time: datetime | None = None, raw_message: str | None = None, service_name: str | None = None, *, is_permanent: bool = False, account_label: str | None = None, usage_limit_scope: UsageLimitScope | None = None, invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED, continuation: Any | None = None) -> None
TransientAgentError(message: str = "", status_code: int | None = None) -> None
RetryableProviderFailureError(message: str = "", *, status_code: int | None = None, service_name: str, classification: str | None = None, observations: tuple[ProviderErrorObservation, ...] = (), invocation_progress: InvocationProgress = InvocationProgress.NOT_STARTED, continuation: Any | None = None) -> None
HardAgentError(message: str = "", status_code: int | None = None, service_name: str = "", classification: str | None = None, observations: tuple[ProviderErrorObservation, ...] = ()) -> None
AgentCredentialFailureError(message: str = "", *, status_code: int | None = None, service_name: str, classification: str | None = None, observations: tuple[ProviderErrorObservation, ...]) -> None
AgentFailedError(invocation_role: str, worktree_path: Path, namespace: str = "", failure_class: str = "", service_name: str = "", provider_session_path: str | None = None, session_root: str = "") -> None
session_dir -> str
```

Exceptional consumer-facing failures include malformed runtime inputs, credential or configuration problems, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions.
