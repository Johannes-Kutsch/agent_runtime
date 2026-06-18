# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts that can be consumed by an application adapter without importing the application itself.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | The reusable runtime package and its stable core public surface. |
| `Runtime Public Surface` | The documented stability surface made of runtime consumer entrypoints and focused adapter seams, not every importable runtime symbol. |
| `Runtime Compatibility Alias` | A transitional older runtime spelling or import path kept only to smooth pre-release migration and not part of the Runtime Public Surface. |
| `Runtime Consumer Surface` | The entrypoint surface intended for ordinary consuming projects that execute prepared agent work without implementing runtime or provider adapters. |
| `Advanced Focused Seam` | A documented runtime public seam for consumers or adapter authors assembling service selection, session planning, provider output, or log lifecycle behavior directly. |
| `Runtime Adapter Seam` | A focused runtime seam implemented by adapter authors who connect the runtime to provider or application infrastructure. |
| `StageSelection` | A single stage selection node containing service, model, effort, and optional fallback. |
| `ServiceName` | A path-safe runtime service identity used for selection, provider state paths, logs, and diagnostics. |
| `ServiceRegistry` | The runtime-owned resolver that maps configured services and stage chains to an executable candidate. |
| `ExecutionProvider` | The focused protocol implemented by provider adapters for execution behavior. |
| `RunKind` | The runtime mode for a service invocation, such as fresh or resumable. |
| `ToolPolicyProfile` | A provider-neutral runtime description of coarse tool-access policy used by provider adapters to render provider-specific command flags. |
| `Tool-less Run` | A runtime invocation whose provider tool access is explicitly forbidden rather than left to provider defaults. |
| `ToolAccess` | A closed runtime value describing whether an invocation has no tools or workspace-backed tool access. |
| `UsageLimitScope` | A caller-defined, validated grouping key used for usage-limit continuation policy. |
| `ProviderSessionState` | The provider-owned session state that records how a run should start or resume. |
| `ProviderSessionId` | The external provider or tool session identifier associated with a runtime service invocation. |
| `ProviderSessionAdapter` | The narrow adapter seam that owns provider-specific session policy. |
| `SessionIntent` | The caller's pre-run declaration of whether an invocation should prepare provider-session continuity or remain ephemeral. |
| `Ephemeral Run` | A runtime invocation that does not prepare or promise provider-session continuity. |
| `Start Session Run` | A runtime invocation that selects a service and prepares provider-session continuity for future invocations. |
| `Resume Session Run` | A runtime invocation that continues an existing provider-session continuity chain without service fallback or reselection. |
| `SessionRunResult` | The completed result value for session-backed runtime execution, whether the session is newly started or resumed. |
| `SessionRuntimeMetadata` | Runtime metadata for completed session-backed execution. |
| `Continuation` | A runtime value containing all consumer-owned data needed to resume a provider-session continuity chain across process calls, including selected service, model, effort, tool access, and provider resume state. |
| `ProviderResumeState` | Provider-owned JSON-compatible data carried inside a continuation and interpreted by the provider adapter when resuming. |
| `InvocationProgress` | Two-state runtime outcome metadata indicating whether the model showed activity before an interruption, such as reasoning, messages, or tool invocation; unknown progress is treated as not started. |
| `RuntimeOutcome` | A canonical runtime result category for expected orchestration outcomes such as completion, usage limits, cancellation, timeout, temporary service unavailability, or confidently retryable provider failure. |
| `SessionNamespace` | An optional path-safe label that partitions provider session state for an invocation role. |
| `WorkInvocation` | The runtime-owned work lifecycle that turns caller intent plus execution dependencies into a text result. |
| `InvocationRole` | A caller-defined, path-safe runtime invocation label used for provider execution metadata, not a runtime-owned workflow model. |
| `AgentRuntimeError` | The base error for runtime failures. |

## Boundary Rules

- The runtime package must remain importable without application modules.
- Application-specific prompt rendering, CLI wiring, issue orchestration, and output parsing belong outside the runtime boundary.
- The runtime/request seam stays a single vertical flow from caller intent through session planning to work invocation.
- The package root should stay a narrow compatibility entrypoint, not a catch-all export surface.
- The documented Runtime Public Surface is a stability promise rather than an inventory of every importable runtime symbol.
- Removing runtime compatibility aliases does not move lifecycle runtime entrypoints to the package root.
- Runtime Compatibility Aliases are not Runtime Public Surface promises.
- Runtime entrypoints should be canonical per mode rather than duplicated across equivalent facades.
- Removed runtime-level compatibility aliases should not remain as alternate behavior paths behind private reachability.
- Removed runtime compatibility aliases should fail on direct import and module attribute access, not merely disappear from documented export lists.
- Removed runtime compatibility aliases should not be preserved in a migration shim namespace.
- Pre-release runtime compatibility aliases may be removed immediately when accepted cleanup requires strict absence.
- Lifecycle-specific runtime execution adapter names are canonical public spellings even when they share the same underlying adapter protocol.
- Request-construction compatibility spellings may remain when they do not create alternate public type names or lifecycle entrypoints.
- Ordinary consuming projects should use runtime entrypoints and adapter seams rather than low-level work invocation internals.
- Runtime-owned selection, availability, and resumability policy stay in the runtime boundary.
- Session continuity and tool access are independent runtime concerns.
- Provider-session continuity is a pre-run intent, not a post-run side effect.
- Ephemeral execution means the runtime does not intentionally prepare provider-session continuity, not merely that the caller discards continuation state.
- The runtime returns continuation state; consuming projects own persistence and retention decisions for that state.
- Continuations are portable but semantically immutable runtime data from the consumer's perspective.
- Continuations may carry provider-owned serializable resume state that the runtime transports but does not interpret.
- Session-backed runtime results return the latest continuation needed for the next resume.
- The runtime must not own durable provider-session storage or cleanup policy.
- Fallback service selection can start a continuity chain but must not silently replace an existing provider-session continuity chain.
- Resumed-session availability or usage-limit failures do not invalidate the continuation and must not trigger automatic fallback.
- Session-backed interruptions should report invocation progress so callers can choose retry or continuation prompts.
- Provider adapters may explicitly report invocation progress, while runtime-owned event reduction may infer progress from known provider events.
- Invocation progress is runtime-wide failure metadata; only session-backed invocations can pair it with continuation state.
- Expected interruption outcomes use two-state invocation progress: started or not started.
- Runtime errors remain classified by failure cause, with interruption progress attached as metadata where relevant.
- Usage limits, cancellation, timeout, temporary service unavailability, and confidently retryable provider failures are normal runtime outcomes at canonical entrypoints rather than exceptional failures.
- Documentation for lifecycle entrypoints should teach expected interruption outcomes through `RuntimeOutcome` before describing lower-level exception classes.
- Cancellation outcomes represent caller- or user-initiated cancellation, not provider-side aborts.
- Invalid service references or malformed service registry configuration remain exceptional failures.
- Credential failures, runtime configuration errors, hard provider failures, adapter/protocol bugs, unclassified provider failures, and unexpected exceptions remain exceptional failures.
- A new continuation becomes meaningful only after provider work has started, not merely after provider session allocation.
- Resumed-session execution keeps service and tool access fixed while defaulting model and effort from the continuation and allowing explicit model or effort overrides.
- Canonical runtime entrypoints should be named around session lifecycle: ephemeral execution, new-session execution, and resumed-session execution.
- With lifecycle-specific entrypoints, session intent is expressed by the entrypoint rather than a defaulted request field.
- The lifecycle entrypoints replace the previous one-shot/resumable canonical API split rather than layering over it.
- Session-backed result and metadata names should use session vocabulary rather than the older resumable runtime-mode vocabulary.
- SessionRunResult does not imply ephemeral selection diagnostics such as selected stage path or fallback use.
- Shared session-backed results should have one canonical public name rather than lifecycle-specific aliases.
- Advanced provider-session planning and provider-capability seams may retain resumable vocabulary when they describe provider resumability rather than canonical lifecycle entrypoints.
- Tool-less execution means provider tool access is explicitly forbidden at the runtime boundary.
- A provider-session continuity chain keeps the same tool policy for its lifetime.
- Resumed-session execution derives tool policy from the resumed continuity state rather than accepting a caller override.
- Workspace access belongs to tool-access configuration rather than session lifecycle.
- Runtime-facing tool settings should be closed tool-access values rather than a loose tool-policy enum plus unrelated workspace fields.
- Canonical runtime requests should represent session intent and tool access as explicit mode values rather than nullable optional fields.
- StageSelection is the canonical stage-chain value; StageOverride is retired compatibility vocabulary.
- Provider execution behavior stays behind focused adapter contracts.
- Provider-specific session details must stay behind explicit adapter contracts.
- Work invocation dependencies should stay focused on execution intent rather than presentation or orchestration concerns.
- Runtime-owned public names should be neutral and caller-supplied where paths or log roots are involved.

## Runtime Surfaces

- Ephemeral prompt execution for already-rendered prompts.
- Session-backed lifecycle execution for provider-backed continuations.
- Caller intent through session planning and work invocation remains one vertical flow.
- Package-root imports stay narrow while behaviorful entrypoints live under focused modules.
- Service selection across nested `StageSelection` chains.
- Provider execution behind adapter contracts.
- Provider session planning and state recovery.
- Provider-session mutation stays behind the provider-facing seam rather than the plan value.
- Text-output reduction from parsed provider events.
- Agent log reservation and append/update lifecycle.

## Flagged Ambiguities

- "Resumable" previously named the provider-session-backed runtime mode; resolved: lifecycle APIs use **Start Session Run** and **Resume Session Run**, shared completed values use **SessionRunResult** and **SessionRuntimeMetadata**, and resumable vocabulary remains acceptable for provider resumability capabilities or lower-level session planning seams.
- "One-shot" previously named standalone single-prompt execution; resolved: public and domain language should use **Ephemeral Run** for execution without provider-session continuity, while one-shot remains only historical compatibility vocabulary.
- "StageOverride" previously named the stage-chain value; resolved: use **StageSelection**.
