# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts consumed by application adapters without importing the application.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | Reusable runtime package and stable core public surface. |
| `Runtime Public Surface` | Documented stability surface: runtime consumer entrypoints, runtime value objects, and built-in provider selection, not every importable symbol. |
| `Runtime Compatibility Alias` | Transitional older runtime spelling or import path kept only for pre-release migration, not Runtime Public Surface. |
| `Runtime Consumer Surface` | Ordinary consuming-project entrypoints for executing prepared agent work without implementing runtime or provider adapters. |
| `Advanced Focused Seam` | Non-consumer seam for maintainers assembling built-in service selection, session planning, provider output, or log lifecycle behavior directly. |
| `Runtime Adapter Seam` | Internal runtime seam implemented by built-in provider integrations, not a consumer-defined service extension point. |
| `Built-in Provider Adapter` | Runtime-shipped provider integration for Claude, Codex, or OpenCode that ordinary consumers select but do not import or implement. |
| `Built-in Execution Substrate` | Runtime-owned mechanism that runs built-in provider commands without application-owned provider services. |
| `Built-in Provider Invocation Seam` | Internal seam behind `RuntimeClient` that executes one prepared built-in provider command and returns observable invocation facts. |
| `ProviderAuth` | Immutable caller-supplied runtime credential data for built-in provider execution. |
| `ClaudeCodeOAuthToken` | Claude Code OAuth token supplied to the built-in Claude provider integration. |
| `StageSelection` | Single stage selection node containing service, model, effort, and optional fallback. |
| `ServiceName` | Path-safe runtime service identity used for selection, invocation records, and diagnostics. |
| `ServiceRegistry` | Runtime-owned resolver that maps built-in services and stage chains to an executable candidate. |
| `ExecutionProvider` | Internal execution contract implemented by runtime-shipped provider integrations. |
| `RunKind` | Runtime mode for a service invocation, such as fresh or resumable. |
| `ProviderInvocationRequest` | Private value carrying command, invocation directory, environment, prompt policy, `RunKind`, optional `ProviderSessionId`, and diagnostics metadata. |
| `ToolPolicy` | Closed public value describing allowed provider tools: `NONE`, `INSPECT_ONLY`, `NO_FILE_MUTATION`, or `UNRESTRICTED`. |
| `ToolPolicyProfile` | Internal provider-neutral adapter policy used to render provider-specific command flags. |
| `Tool-less Run` | Runtime invocation whose `ToolPolicy` explicitly forbids provider tools. |
| `ToolAccess` | Retired target vocabulary for the public API; use `ToolPolicy`. |
| `Invocation Directory` | Host directory where runtime launches a provider command; public request field is `invocation_dir`. |
| `Tool Workspace` | Invocation Directory when runtime exposes it through provider tools. |
| `UsageLimitScope` | Transitional caller-defined grouping key; usage-limit grouping belongs outside core runtime API. |
| `ProviderSessionState` | Provider-owned session state recording how a run should start or resume. |
| `ProviderSessionId` | External provider or tool session identifier associated with a runtime service invocation. |
| `ProviderInvocationResult` | Private provider invocation result containing normalized output, optional `ProviderUsage`, raw stdout when lifecycle policy needs it, and optional `ProviderSessionId`. |
| `ProviderSessionAdapter` | Internal provider-session seam owning built-in provider session policy. |
| `SessionIntent` | Caller pre-run declaration that an invocation should prepare provider-session continuity or remain ephemeral. |
| `Ephemeral Run` | Runtime invocation that does not prepare or promise provider-session continuity. |
| `Start Session Run` | Runtime invocation that selects a service and prepares provider-session continuity. |
| `Resume Session Run` | Runtime invocation that continues an existing provider-session continuity chain without fallback or reselection. |
| `Session-backed Provider` | Built-in provider that can produce and consume portable continuation data. |
| `SessionRunResult` | Completed result for session-backed execution, containing output text and meaningful continuation. |
| `SessionRuntimeMetadata` | Runtime metadata for completed session-backed execution. |
| `Continuation` | Opaque portable resume token callers persist and pass back to resume a continuity chain. |
| `ProviderResumeState` | Provider-owned opaque data carried inside a continuation and interpreted only by provider adapter. |
| `RuntimeStateDir` | Transitional caller-supplied root previously used for provider-native session state; active session-backed requests do not require it. |
| `RuntimeLogsDir` | Transitional caller-supplied root previously used for runtime invocation logs; callers now own durable trace persistence. |
| `InvocationRecord` | Structured runtime output describing an invocation for caller persistence or display. |
| `Live Runtime Output` | Provider-neutral observable agent-message text and selected service identity emitted during runtime invocation. |
| `AgentMessageTurn` | Immutable provider-neutral unit with `text` and selected `service_name` for one assistant-authored message. |
| `RuntimeClient` | Caller-owned runtime object holding in-process built-in provider availability state without durable provider session storage. |
| `InvocationProgress` | Two-state interruption metadata indicating whether model activity was observed; unknown progress means not started. |
| `RuntimeOutcome` | Canonical result category for expected orchestration outcomes: completion, usage limits, cancellation, timeout, temporary unavailability, or retryable provider failure. |
| `Live Provider Smoke Test` | Opt-in validation run outside default tests that exercises real built-in providers through Runtime Public Surface. |
| `ProviderUsage` | Provider-reported usage metadata: input/output tokens, cache-read/cache-creation input tokens, optional USD cost, and optional provider duration. |
| `SessionNamespace` | Transitional secondary label formerly used to partition runtime-managed provider session state; active session-backed requests do not require it. |
| `WorkInvocation` | Runtime-owned work lifecycle that turns caller intent plus execution dependencies into a text result. |
| `InvocationRole` | Obsolete transitional term for caller-defined invocation label; core runtime requests should not require caller-defined labels. |
| `AgentRuntimeError` | Base error for runtime failures. |

## Boundary Rules

- Runtime package must remain importable without application modules.
- Application prompt rendering, CLI wiring, issue orchestration, output parsing, display, redaction, durable trace persistence, and retention belong outside runtime boundary.
- Runtime Public Surface is documented stability promise, not inventory of importable symbols.
- Runtime Consumer Surface uses `RuntimeClient`, lifecycle requests, public outcome/result values, `StageSelection`, `ProviderAuth`, `Continuation`, `InvocationRecord`, and `ToolPolicy`.
- Ordinary consumers select built-in providers through runtime call arguments, not provider services, service registries, execution adapters, provider-session adapters, command builders, provider event parsers, or provider DTO streams.
- Consumer-defined provider services are not supported runtime functionality.
- `RuntimeClient` owns in-process provider availability/exhaustion state, is safe for concurrent reuse, and owns no durable provider state or logs.
- Provider selection remains caller-supplied through `StageSelection`; runtime validates built-in service, model, and effort values.
- Built-in provider credentials are per-request `ProviderAuth`; missing explicit credentials are credential failures and stop fallback.
- Runtime-owned selection, availability, resumability, failure classification, path-safety validation, and provider parsing stay inside runtime boundary.
- Built-in Provider Invocation Seam is internal and must not become a consumer-defined adapter extension point.
- Provider event DTOs, provider-specific session details, command rendering, stream parsing, and provider flag profiles are internal.
- Public provider failure diagnostics may expose provider observations; consumers own storage, display, and redaction.
- Live Runtime Output is per-request observation of `AgentMessageTurn` values, not arbitrary provider chunks, token streaming, replay, logs, or alternate lifecycle entrypoints.
- Live Runtime Output is independent of session lifecycle and `ToolPolicy`; completed runtime output remains authoritative.
- Live Runtime Output observers are synchronous, notification-only, at-most-once per provider attempt, and consumer-owned for async bridging and backpressure.
- Live Runtime Output callback failures propagate as exceptional consumer failures.
- Session continuity and tool policy are independent runtime concerns.
- Ephemeral execution does not intentionally prepare provider-session continuity.
- Start Session Run returns continuation state; callers own persistence and retention.
- Resume Session Run continues an existing continuity chain without fallback or reselection.
- Continuations are opaque, portable, semantically immutable, and may carry provider-owned serializable resume state.
- Session-backed execution is limited to built-in providers that produce and consume portable continuation data.
- Runtime must not own durable provider-session storage, durable invocation-log storage, or cleanup policy.
- Resumed-session availability or usage-limit failures do not invalidate continuations and must not trigger automatic fallback.
- Session-backed interruptions report invocation progress so callers can choose retry or continuation prompts.
- Usage limits, cancellation, timeout, temporary unavailability, and confidently retryable provider failures are normal `RuntimeOutcome` values.
- Credential failures, runtime configuration errors, hard provider failures, adapter/protocol bugs, unclassified provider failures, invalid service references, and unexpected exceptions remain exceptional failures.
- Provider-reported usage belongs on runtime outcomes whenever observed, including interrupted outcomes.
- Cancellation and timeout outcomes report only usage observed before interruption; runtime does not perform provider-specific post-kill usage recovery.
- A new continuation becomes meaningful only after provider work has started.
- Resumed-session execution keeps service and `ToolPolicy` fixed while defaulting model and effort from continuation and allowing explicit model/effort overrides.
- Runtime requests require explicit `ToolPolicy`; non-`NONE` policies grant Invocation Directory as Tool Workspace.
- Invocation Directory and Tool Workspace are distinct permission concepts, but not separate public paths in current runtime model.
- `ToolPolicyProfile` is not ordinary consumer-facing API.
- StageSelection is canonical stage-chain value; StageOverride is retired compatibility vocabulary.
- Runtime Compatibility Aliases are not Runtime Public Surface promises and may be removed before release.
- Lifecycle-specific runtime execution adapter names are canonical public spellings even when they share adapter protocols.
- Live Provider Smoke Tests are opt-in maintainer tooling, not default automated tests or Runtime Public Surface additions.
- Live Provider Smoke Tests prove real provider invocation through Runtime Public Surface; they do not judge answer quality, tool usefulness, or strict instruction following.

## Runtime Surfaces

- Ephemeral prompt execution for already-rendered prompts.
- Session-backed lifecycle execution for provider-backed continuations.
- Caller intent through session planning and work invocation as one vertical flow.
- Narrow package-root imports while behaviorful entrypoints live under focused modules.
- Service selection across nested `StageSelection` chains.
- Built-in provider execution behind runtime-owned internal adapter contracts.
- Provider session planning and state recovery.
- Text-output reduction from parsed provider events.
- Invocation-record production for callers that want durable traces.

## Flagged Ambiguities

- "Resumable": use **Start Session Run** and **Resume Session Run** for lifecycle APIs; keep resumable only for provider capabilities or lower-level planning.
- "One-shot": use **Ephemeral Run** for execution without provider-session continuity.
- "StageOverride": use **StageSelection**.
- "OpenAI" in issue #93 meant **OpenCode**.
- "Claude API key": use **ClaudeCodeOAuthToken**, not generic Anthropic API key.
- "Adapter author": custom provider services are not supported runtime extension point.
- "worktree": use **Invocation Directory** for command location and **Tool Workspace** for tool access to that directory.
- `ToolAccess`: use **ToolPolicy**.
- `ToolPolicy.RESTRICTED`, `ToolPolicy.PARTIAL`, and `ToolPolicy.FULL`: use `ToolPolicy.INSPECT_ONLY`, `ToolPolicy.NO_FILE_MUTATION`, and `ToolPolicy.UNRESTRICTED`.
