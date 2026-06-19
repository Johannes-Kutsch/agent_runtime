# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts that can be consumed by an application adapter without importing the application itself.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | The reusable runtime package and its stable core public surface. |
| `Runtime Public Surface` | The documented stability surface made of runtime consumer entrypoints, runtime value objects, and built-in provider selection, not every importable runtime symbol. |
| `Runtime Compatibility Alias` | A transitional older runtime spelling or import path kept only to smooth pre-release migration and not part of the Runtime Public Surface. |
| `Runtime Consumer Surface` | The entrypoint surface intended for ordinary consuming projects that execute prepared agent work without implementing runtime or provider adapters. |
| `Advanced Focused Seam` | A non-consumer runtime seam for maintainers assembling built-in service selection, session planning, provider output, or log lifecycle behavior directly. |
| `Runtime Adapter Seam` | An internal runtime seam implemented by built-in provider integrations, not a supported extension point for consumer-defined services. |
| `Built-in Provider Adapter` | A runtime-shipped provider integration for a supported external agent provider such as Claude, Codex, or OpenCode that ordinary consumers select but do not import or implement. |
| `Built-in Execution Substrate` | The runtime-owned mechanism that runs built-in provider commands for ordinary consumers without application-owned provider services. |
| `Built-in Provider Invocation Seam` | Internal runtime seam behind `RuntimeClient` that executes one prepared built-in provider command and returns observable invocation facts such as normalized output, provider usage, raw provider stdout lines, and observed `ProviderSessionId`; not a Runtime Public Surface or consumer-defined adapter point. |
| `ProviderAuth` | Immutable caller-supplied runtime credential data for built-in provider execution. |
| `ClaudeCodeOAuthToken` | The Claude Code OAuth token supplied to the built-in Claude provider integration. |
| `StageSelection` | A single stage selection node containing service, model, effort, and optional fallback. |
| `ServiceName` | A path-safe runtime service identity used for selection, provider state paths, logs, and diagnostics. |
| `ServiceRegistry` | The runtime-owned resolver that maps built-in services and stage chains to an executable candidate. |
| `ExecutionProvider` | The internal execution contract implemented by runtime-shipped provider integrations. |
| `RunKind` | The runtime mode for a service invocation, such as fresh or resumable. |
| `ProviderInvocationRequest` | Private value describing all facts required by the Built-in Provider Invocation Seam: command, invocation directory, environment, prompt content or prompt-path policy, `RunKind`, invocation role, usage-limit scope, log context, and optional `ProviderSessionId`. |
| `ToolPolicy` | A closed public runtime value describing what provider tools may be used during an invocation: `NONE`, `INSPECT_ONLY`, `NO_FILE_MUTATION`, or `UNRESTRICTED`. |
| `ToolPolicyProfile` | Internal provider-neutral adapter policy used by provider adapters to render provider-specific command flags. |
| `Tool-less Run` | A runtime invocation whose `ToolPolicy` explicitly forbids provider tool access rather than leaving tools to provider defaults. |
| `ToolAccess` | Retired target vocabulary for the public API; use `ToolPolicy` instead. |
| `Invocation Directory` | The host directory where the runtime launches a provider command; the canonical public request field is `invocation_dir`. |
| `Tool Workspace` | The Invocation Directory when the runtime is allowed to expose it through provider tools. |
| `UsageLimitScope` | Transitional caller-defined grouping key for usage-limit continuation policy; usage-limit grouping belongs outside the core runtime API. |
| `ProviderSessionState` | The provider-owned session state that records how a run should start or resume. |
| `ProviderSessionId` | The external provider or tool session identifier associated with a runtime service invocation. |
| `ProviderInvocationResult` | Private value returned by the Built-in Provider Invocation Seam containing normalized output, optional `ProviderUsage`, raw stdout lines when lifecycle policy needs them, and optional observed `ProviderSessionId`. |
| `ProviderSessionAdapter` | The internal provider-session seam that owns built-in provider session policy. |
| `SessionIntent` | The caller's pre-run declaration of whether an invocation should prepare provider-session continuity or remain ephemeral. |
| `Ephemeral Run` | A runtime invocation that does not prepare or promise provider-session continuity. |
| `Start Session Run` | A runtime invocation that selects a service and prepares provider-session continuity for future invocations. |
| `Resume Session Run` | A runtime invocation that continues an existing provider-session continuity chain without service fallback or reselection. |
| `Session-backed Provider` | A built-in provider that can produce and consume portable continuation data. |
| `SessionRunResult` | The completed result value for session-backed runtime execution, containing output text and a meaningful continuation. |
| `SessionRuntimeMetadata` | Runtime metadata for completed session-backed execution. |
| `Continuation` | An opaque portable resume token that callers persist and pass back to resume a provider-session continuity chain. |
| `ProviderResumeState` | Provider-owned opaque data carried inside a continuation and interpreted only by the provider adapter when resuming. |
| `RuntimeStateDir` | Transitional caller-supplied directory root where built-in provider integrations keep provider-native session state for session-backed runs. |
| `RuntimeLogsDir` | Transitional caller-supplied directory root where runtime invocation logs are written. |
| `InvocationRecord` | Structured runtime output describing an invocation for callers that want to persist or display execution traces. |
| `Live Runtime Output` | Provider-neutral consumer-observable agent-message text and selected service identity emitted during a runtime invocation, not token-by-token streaming, derived from runtime-owned provider event parsing without exposing provider-specific event DTOs, raw JSON, or stdout as the public contract. |
| `AgentMessageTurn` | An immutable provider-neutral unit with `text` and selected `service_name`, observed by the runtime as one meaningful assistant-authored message during provider execution. |
| `RuntimeClient` | Caller-owned runtime object that holds in-process built-in provider availability state across calls without owning durable provider session storage. |
| `InvocationProgress` | Two-state runtime outcome metadata indicating whether the model showed activity before an interruption, such as reasoning, messages, or tool invocation; unknown progress is treated as not started. |
| `RuntimeOutcome` | A canonical runtime result category for expected orchestration outcomes such as completion, usage limits, cancellation, timeout, temporary service unavailability, or confidently retryable provider failure. |
| `Live Provider Smoke Test` | An opt-in validation run that exercises real built-in provider integrations outside the default test suite. |
| `ProviderUsage` | Provider-reported usage metadata for a runtime invocation: input tokens, output tokens, cache-read input tokens, cache-creation input tokens, optional cost in USD, and optional provider duration in seconds. |
| `SessionNamespace` | Transitional secondary path-safe label that further partitions runtime-managed provider session state. |
| `WorkInvocation` | The runtime-owned work lifecycle that turns caller intent plus execution dependencies into a text result. |
| `InvocationRole` | Obsolete transitional term for a caller-defined invocation label; core runtime requests should not require caller-defined labels. |
| `AgentRuntimeError` | The base error for runtime failures. |

## Boundary Rules

- The runtime package must remain importable without application modules.
- Application-specific prompt rendering, CLI wiring, issue orchestration, and output parsing belong outside the runtime boundary.
- Application-specific protocol parsing of Live Runtime Output belongs outside the runtime boundary.
- The runtime/request seam stays a single vertical flow from caller intent through session planning to work invocation.
- The package root should stay a narrow compatibility entrypoint, not a catch-all export surface.
- `AgentMessageTurn` is shared public runtime vocabulary and may be exported from both the package root and `agent_runtime.runtime`.
- The documented Runtime Public Surface is a stability promise rather than an inventory of every importable runtime symbol.
- External provider adapter seams should be removed from the documented Runtime Public Surface.
- Provider event DTOs are internal built-in adapter details, not consumer-facing API.
- Live Runtime Output is an observation channel for `AgentMessageTurn` text, not arbitrary provider output chunks; completed runtime output remains the authoritative invocation result.
- Live Runtime Output callback failures are consumer-side failures and should not be silently swallowed by the runtime.
- Live Runtime Output callback failures propagate as exceptional consumer failures rather than runtime interruption outcomes.
- Consumers observe Live Runtime Output through per-request invocation observers rather than RuntimeClient-wide display state or alternate streaming lifecycle entrypoints.
- The public per-request observer should use Live Runtime Output vocabulary, such as `on_live_output`, rather than provider- or parser-specific naming.
- Live Runtime Output observers are synchronous; async consumers bridge observation into their own queues or event loops outside the runtime boundary.
- Backpressure from synchronous Live Runtime Output observers is consumer responsibility; the runtime does not own buffering or drop policy for observed turns.
- Live Runtime Output observers are notification-only and do not steer runtime control flow; consumers use cancellation to stop an invocation.
- Live Runtime Output observers receive `AgentMessageTurn` values directly, not bare strings, speculative event wrappers, or provider-specific event payloads.
- Live Runtime Output values include the selected `ServiceName` so consumers can correlate observed turns during provider fallback or display.
- Live Runtime Output values do not repeat selected model or effort metadata unless a separate live-display need is established.
- Live Runtime Output may include turns from provider attempts later abandoned by fallback; consumers use `ServiceName` and the authoritative final output to correlate or discard those observations.
- Live Runtime Output observers should see each runtime-observed `AgentMessageTurn` at most once per provider attempt.
- Live Runtime Output is independent of session lifecycle and may be observed for Ephemeral Run, Start Session Run, and Resume Session Run invocations.
- Live Runtime Output reports only newly observed output from the current invocation and does not replay prior turns from continuations, logs, or provider transcript state.
- Live Runtime Output is not coupled to `ProviderSessionId` detection and should not delay observed turns until session metadata is available.
- Live Runtime Output carries normalized agent-message text without runtime-owned durable redaction or persistence policy; consumers own display, redaction, and persistence decisions.
- Live Runtime Output uses the same runtime-owned provider parsing semantics as final output reduction, but observed turns are not guaranteed to be literal substrings of authoritative completed output.
- Live Runtime Output does not change completed or interrupted `RuntimeOutcome` output semantics.
- Live Runtime Output observation is independent of `ToolPolicy` and grants no tool capability, workspace access, or logging permission.
- Built-in Provider Adapters should emit Live Runtime Output when runtime-owned provider parsing observes an `AgentMessageTurn` equivalent; absence of live turns for a provider must not reject an invocation.
- Public provider failure errors may expose provider diagnostic observations; consumers own storage, display, and redaction policy for those diagnostics.
- Removing runtime compatibility aliases does not move lifecycle runtime entrypoints to the package root.
- Runtime Compatibility Aliases are not Runtime Public Surface promises.
- Runtime entrypoints should be canonical per mode rather than duplicated across equivalent facades.
- Removed runtime-level compatibility aliases should not remain as alternate behavior paths behind private reachability.
- Removed runtime compatibility aliases should fail on direct import and module attribute access, not merely disappear from documented export lists.
- Removed runtime compatibility aliases should not be preserved in a migration shim namespace.
- Pre-release runtime compatibility aliases may be removed immediately when accepted cleanup requires strict absence.
- Lifecycle-specific runtime execution adapter names are canonical public spellings even when they share the same underlying adapter protocol.
- Request-construction compatibility spellings may remain when they do not create alternate public type names or lifecycle entrypoints.
- Ordinary consuming projects should use runtime entrypoints rather than low-level work invocation internals or provider adapter seams.
- Ordinary consuming projects may observe Live Runtime Output through the Runtime Consumer Surface without importing provider adapters or provider event parsers.
- Ordinary consuming projects should select Built-in Provider Adapters through runtime call arguments rather than constructing provider services or service registries.
- Consumer-defined provider services are not supported runtime functionality.
- Provider selection remains caller-supplied through `StageSelection`; the runtime validates and executes built-in service names but does not own application workflow default chains.
- Built-in service, model, and effort values are validated by the runtime before provider execution.
- Lifecycle runtime constructors should not expose execution adapter or service registry injection on the consumer API.
- Ordinary runtime execution uses a Built-in Execution Substrate; application-owned Docker orchestration, dependency installation, managed worktrees, and preflight setup remain outside the runtime boundary.
- Built-in provider subprocess dispatch, prompt-file lifecycle, and invocation-log interaction belong behind the Built-in Provider Invocation Seam, while provider-specific command rendering and lifecycle policy remain runtime-owned internals.
- The Built-in Provider Invocation Seam is internal and must not become a consumer-defined provider adapter extension point.
- Built-in provider credentials are supplied through per-request `ProviderAuth` data rather than process-global runtime setup.
- `ProviderAuth` only needs credentials for explicit-credential providers reachable from the request's `StageSelection` chain; Codex uses host auth state.
- Missing built-in provider credentials are credential failures, not malformed runtime configuration.
- Built-in provider credential failures stop execution rather than falling through to stage fallback.
- Runtime-owned selection, availability, and resumability policy stay in the runtime boundary.
- Built-in provider availability and exhaustion state live in the caller-owned `RuntimeClient`, not in process globals or consumer-created service objects.
- `RuntimeClient` is safe to reuse across concurrent runtime requests and synchronizes built-in provider availability updates internally.
- Built-in Provider Adapters are runtime-owned provider integrations, not application orchestration.
- Built-in Provider Adapter internals are not ordinary consumer API even though the runtime distribution ships them.
- Session continuity and tool access are independent runtime concerns.
- Provider-session continuity is a pre-run intent, not a post-run side effect.
- Ephemeral execution means the runtime does not intentionally prepare provider-session continuity, not merely that the caller discards continuation state.
- The runtime returns continuation state; consuming projects own persistence and retention decisions for that state.
- Continuations are opaque, portable, and semantically immutable runtime data from the consumer's perspective.
- Continuations may carry provider-owned serializable resume state, including encoded provider state when needed.
- Continuations do not expose provider state paths as their public resume contract.
- Session-backed runtime results return the latest continuation needed for the next resume.
- Session-backed execution is available only for built-in provider integrations that can produce and consume portable continuation data.
- Session-backed built-in provider execution must preserve provider-native transcript continuity through portable continuations.
- Session-backed public requests do not require runtime-managed provider state directories.
- The runtime must not own durable provider-session storage, durable invocation-log storage, or cleanup policy.
- Durable invocation traces are returned as structured records for callers to persist when needed.
- Fallback service selection can start a continuity chain but must not silently replace an existing provider-session continuity chain.
- Resumed-session availability or usage-limit failures do not invalidate the continuation and must not trigger automatic fallback.
- Session-backed interruptions should report invocation progress so callers can choose retry or continuation prompts.
- Built-in Provider Adapters may explicitly report invocation progress, while runtime-owned event reduction may infer progress from known provider events.
- Invocation progress is runtime-wide failure metadata; only session-backed invocations can pair it with continuation state.
- Expected interruption outcomes use two-state invocation progress: started or not started.
- Provider-reported usage metadata belongs on runtime outcomes whenever the provider reports it, including interrupted outcomes.
- `RuntimeOutcome` carries `ProviderUsage` as top-level optional outcome metadata.
- Cached provider input tokens map to `ProviderUsage.cache_read_input_tokens` when the provider semantics are cached prompt/input tokens.
- Built-in provider parsers should emit rich `ProviderUsage` events for usage reporting rather than treating prompt-token-only events as the usage contract.
- Cancellation and timeout outcomes report only provider usage observed before interruption; the runtime does not perform provider-specific post-kill usage recovery.
- Runtime errors remain classified by failure cause, with interruption progress attached as metadata where relevant.
- Usage limits, cancellation, timeout, temporary service unavailability, and confidently retryable provider failures are normal runtime outcomes at canonical entrypoints rather than exceptional failures.
- Documentation for lifecycle entrypoints should teach expected interruption outcomes through `RuntimeOutcome` before describing lower-level exception classes.
- Cancellation outcomes represent caller- or user-initiated cancellation, not provider-side aborts.
- Invalid service references or malformed service registry configuration remain exceptional failures.
- Credential failures, runtime configuration errors, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions remain exceptional failures.
- A new continuation becomes meaningful only after provider work has started, not merely after provider session allocation.
- Resumed-session execution keeps service and tool access fixed while defaulting model and effort from the continuation and allowing explicit model or effort overrides.
- Canonical runtime entrypoints should be named around session lifecycle: ephemeral execution, new-session execution, and resumed-session execution.
- Ordinary consumer execution should go through a caller-owned `RuntimeClient` with lifecycle methods for ephemeral, new-session, and resumed-session runs.
- With lifecycle-specific entrypoints, session intent is expressed by the entrypoint rather than a defaulted request field.
- The lifecycle entrypoints replace the previous one-shot/resumable canonical API split rather than layering over it.
- Session-backed result and metadata names should use session vocabulary rather than the older resumable runtime-mode vocabulary.
- SessionRunResult does not imply ephemeral selection diagnostics such as selected stage path or fallback use.
- Shared session-backed results should have one canonical public name rather than lifecycle-specific aliases.
- Advanced provider-session planning and provider-capability seams may retain resumable vocabulary when they describe provider resumability rather than canonical lifecycle entrypoints.
- Tool-less execution means provider tool access is explicitly forbidden at the runtime boundary.
- Built-in Provider Adapters accept every public `ToolPolicy` for every supported service/model pair and enforce each policy with the strongest provider-supported mechanism.
- Tool-policy enforcement strength may vary by provider even though every public policy is valid for every supported service/model pair.
- A provider-session continuity chain keeps the same `ToolPolicy` for its lifetime, including `ToolPolicy.NONE`.
- Resumed-session execution derives `ToolPolicy` from the resumed continuity state rather than accepting a caller override.
- Workspace access belongs to tool-access configuration rather than session lifecycle.
- Under the host subprocess execution substrate, the request path is the Invocation Directory, while any non-`NONE` `ToolPolicy` grants that directory as the Tool Workspace.
- Runtime requests require an Invocation Directory even for tool-less execution; tool-less execution does not grant a Tool Workspace.
- The Invocation Directory and Tool Workspace are distinct permission concepts, but not separate public paths in the current runtime model.
- Non-`NONE` tool policies use the Invocation Directory as their Tool Workspace rather than requiring consumers to pass the same path twice.
- Runtime-facing tool settings should be a closed `ToolPolicy` value rather than a loose policy enum plus unrelated workspace fields.
- `ToolPolicy` is the consumer-facing tool contract; provider flag profiles are internal built-in adapter policy.
- `ToolPolicyProfile` is not part of the ordinary consumer-facing public API.
- Runtime requests require an explicit `ToolPolicy`; tool policy must not be inferred from omission.
- `ToolPolicy.INSPECT_ONLY` is an allowlist policy for workspace inspection tools.
- `ToolPolicy.NO_FILE_MUTATION` is a denylist policy that permits tools while forbidding direct workspace file mutation.
- `ToolPolicy.UNRESTRICTED` means the runtime adds no tool restriction beyond provider defaults.
- Canonical runtime requests should represent session intent and tool access as explicit mode values rather than nullable optional fields.
- StageSelection is the canonical stage-chain value; StageOverride is retired compatibility vocabulary.
- Provider execution behavior stays behind runtime-owned internal adapter contracts.
- Provider-specific session details must stay behind runtime-owned internal adapter contracts.
- Work invocation dependencies should stay focused on execution intent rather than presentation or orchestration concerns.
- Runtime-owned public names should be neutral and caller-supplied where paths or log roots are involved.

## Runtime Surfaces

- Ephemeral prompt execution for already-rendered prompts.
- Session-backed lifecycle execution for provider-backed continuations.
- Caller intent through session planning and work invocation remains one vertical flow.
- Package-root imports stay narrow while behaviorful entrypoints live under focused modules.
- Service selection across nested `StageSelection` chains.
- Built-in provider execution behind runtime-owned internal adapter contracts.
- Provider session planning and state recovery.
- Provider-session mutation stays behind the provider-facing seam rather than the plan value.
- Text-output reduction from parsed provider events.
- Agent log reservation and append/update lifecycle.

## Flagged Ambiguities

- "Resumable" previously named the provider-session-backed runtime mode; resolved: lifecycle APIs use **Start Session Run** and **Resume Session Run**, shared completed values use **SessionRunResult** and **SessionRuntimeMetadata**, and resumable vocabulary remains acceptable for provider resumability capabilities or lower-level session planning seams.
- "One-shot" previously named standalone single-prompt execution; resolved: public and domain language should use **Ephemeral Run** for execution without provider-session continuity, while one-shot remains only historical compatibility vocabulary.
- "StageOverride" previously named the stage-chain value; resolved: use **StageSelection**.
- "OpenAI" was used in issue #93 for the third built-in provider; resolved: the intended provider is **OpenCode**, matching the existing pycastle service.
- "Claude API key" was used loosely for Claude credentials; resolved: the migrated Claude provider uses **ClaudeCodeOAuthToken**, not a generic Anthropic API key.
- "Adapter author" previously named an external runtime audience; resolved: custom provider services are not a supported runtime extension point.
- "worktree" has been used for both command launch location and provider tool workspace; resolved: use **Invocation Directory** and public `invocation_dir` for the request-level command location, while non-`NONE` `ToolPolicy` grants that directory as the **Tool Workspace** without a second public path.
- `ToolAccess` previously wrapped no-tools versus workspace-backed tool access; resolved: collapse the public API to **ToolPolicy**, where `ToolPolicy.NONE` forbids provider tools and non-`NONE` policies grant tools against the Invocation Directory.
- `ToolPolicy.RESTRICTED`, `ToolPolicy.PARTIAL`, and `ToolPolicy.FULL` were ambiguous names; resolved: use `ToolPolicy.INSPECT_ONLY`, `ToolPolicy.NO_FILE_MUTATION`, and `ToolPolicy.UNRESTRICTED`.
