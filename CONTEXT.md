# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts consumed by application adapters without importing the application.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | Reusable runtime package and stable core public surface. |
| `Runtime Public Surface` | Documented stability surface: consumer entrypoints, runtime value objects, and built-in provider selection — not every importable symbol. |
| `Runtime Consumer Surface` | Ordinary consuming-project entrypoints for executing prepared agent work without implementing runtime or provider adapters. |
| `Built-in Provider Invocation` | Internal mechanism behind `RuntimeClient` that executes one prepared built-in provider command and returns observable invocation facts. |
| `ProviderAuth` | Immutable credential data carried by `ProviderSelection`, or supplied to Resume Session Run since continuations don't store credentials. |
| `ProviderSelection` | Public request value selecting one service, model, effort, and provider credentials for one invocation. |
| `OpenCode Go Integration` | Built-in `opencode` service integration for OpenCode Go subscription access. |
| `Consumer Fallback` | Consuming-project orchestration choosing whether to start a separate runtime invocation after a prior one completes or fails. |
| `ToolPolicy` | Closed public value of allowed provider tools: `NONE`, `INSPECT_ONLY`, `NO_FILE_MUTATION`, or `UNRESTRICTED`. |
| `Invocation Directory` | Host directory where runtime launches a provider command; request field `invocation_dir`. |
| `Tool Workspace` | Invocation Directory when runtime exposes it through provider tools. |
| `Idle Timeout` | Per-run heartbeat watchdog: max seconds without any observed `Agent Event` before runtime aborts with a `timed_out` outcome. Consumer-configurable per run; default 300s. |
| `Ephemeral Run` | Invocation that neither prepares nor promises provider-session continuity. |
| `Start Session Run` | Invocation that selects a service and prepares provider-session continuity. |
| `Resume Session Run` | Invocation that continues an existing continuity chain without fallback or reselection. |
| `Session-backed Provider` | Built-in provider that can produce and consume portable continuation data. |
| `Continuation` | Opaque portable resume token callers persist and pass back to resume a continuity chain. |
| `InvocationRecord` | Structured finished-run output for caller persistence or display: the complete ordered Agent Event sequence plus terminal metadata and raw provider output evidence. |
| `Live Runtime Output` | Provider-neutral stream of typed `Agent Event` observations emitted during invocation. |
| `Agent Event` | One observed signal, discriminated by type (agent message, agent tool call, or other agent life sign), carrying both the filtered/neutral interpretation and the raw provider output it derived from, plus selected service identity. |
| `RuntimeClient` | Caller-owned runtime object executing requests without durable provider session storage or cross-call provider availability policy. |
| `RuntimeOutcome` | Canonical result category for one invocation: completion, usage limits, cancellation, timeout, selected-provider temporary unavailability, or retryable provider failure. |
| `Live Provider Smoke Test` | Opt-in validation run outside default tests exercising real built-in providers through Runtime Public Surface. |
| `Full Live Smoke Matrix` | Maintainer confidence scope for Live Provider Smoke Tests: all lifecycle modes plus every public `ToolPolicy` for each selected configured provider. |
| `Live Smoke Default` | Cost-first runtime-supported provider/model/effort tuple used by Live Provider Smoke Tests absent CLI or environment override. |
| `ProviderUsage` | Provider-reported usage: input/output tokens, cache-read/cache-creation input tokens, optional USD cost, optional provider duration. |

## Boundary Rules

- Runtime package must import without application modules.
- Application prompt rendering, CLI wiring, issue orchestration, output parsing, display, redaction, durable trace persistence, and retention belong outside the runtime boundary.
- Runtime Consumer Surface uses `RuntimeClient`, lifecycle requests, public outcome/result values, `ProviderSelection`, `ProviderAuth`, `Continuation`, `InvocationRecord`, and `ToolPolicy`.
- Consumers import `ToolPolicy` from public runtime/root modules, not adapter contract modules.
- Ordinary consumers select built-in providers through runtime call arguments, not provider services, registries, execution adapters, provider-session adapters, command builders, event parsers, or DTO streams.
- Consumer-defined provider services are not supported runtime functionality.
- Prompt-runtime execution adapter paths are retired.
- `RuntimeClient` owns no cross-call provider availability policy, durable provider state, or logs.
- Provider selection is caller-supplied via the `provider_selection` request field; runtime validates built-in service, model, effort, and relevant credentials.
- `ProviderSelection` construction validates value shape; invocation validates built-in provider support and availability.
- `ProviderSelection` requires explicit service, model, and effort; selection defaults belong to consumers or maintainer tooling.
- OpenCode Go Integration accepts service-local model ids on `ProviderSelection.model`; provider-specific prefixes such as `opencode-go/` are internal command/config rendering details.
- `ProviderSelection.auth` is optional; selected providers requiring explicit credentials validate the relevant ProviderAuth field during invocation.
- `ProviderSelection` equality includes credentials, but textual representations must not reveal credential values.
- `ProviderAuth` equality includes credential values, but textual representations must redact them.
- Resume Session Run derives provider identity from the continuation and accepts request-time ProviderAuth only for credentials.
- Runtime performs no provider fallback inside a single invocation; fallback is Consumer Fallback across separate invocations.
- Runtime outcomes and exceptions describe one invocation only and do not classify Consumer Fallback eligibility.
- `no_service_available` describes temporary unavailability of the selected provider before model work starts, not exhaustion of a provider chain.
- Unsupported service, model, or effort selections are configuration errors, not `no_service_available` outcomes.
- Normal RuntimeOutcome values identify the selected provider service, model, and effort.
- Runtime results report selected provider facts for one invocation, not Consumer Fallback attempt paths.
- Invocation records describe one invocation and do not carry Consumer Fallback group or attempt identifiers.
- Absence of `InvocationRecord` values means runtime has no provider invocation evidence for that outcome.
- Runtime returns an `InvocationRecord` for each provider dispatch that starts, including interrupted outcomes.
- Built-in provider credentials are part of `ProviderSelection`; missing explicit credentials are credential failures and do not trigger Consumer Fallback inside runtime.
- Runtime-owned selection, availability, resumability, failure classification, path-safety validation, and provider parsing stay inside the runtime boundary.
- Runtime classifies provider failures but never acts on the classification: no waiting, retry scheduling, or in-runtime fallback. Provider-specific detection (e.g. subscription-denial recognition) stays internal; surfaced outcomes carry a neutral category plus the raw provider error.
- Built-in Provider Invocation is internal and must not become a consumer-defined adapter extension or request-time injection point.
- WorkInvocation and execution adapter seams are internal and must not be consumer-accessible.
- Public-looking internal modules should be moved behind underscore-prefixed names before release where practical.
- Retiring a concept means deleting obsolete code where feasible; any internal survival requires a current built-in runtime purpose and must not reappear on documented/root/runtime surfaces.
- Built-in Provider Invocation must use runtime-neutral internal artifact names, not pycastle-specific prompt or session naming.
- Provider event DTOs, provider-specific session details, command rendering, stream parsing, and provider flag profiles are internal.
- Public provider failure diagnostics may expose provider observations; consumers own storage, display, and redaction.
- Live Runtime Output is per-request observation of typed `Agent Event` values, not arbitrary provider chunks, token streaming, replay, logs, or alternate lifecycle entrypoints.
- Agent Events are discriminated by a closed type set — agent message, agent tool call, other agent life sign; consumers branch on type.
- Each Agent Event carries both a filtered/neutral view (typed content, e.g. tool identity and neutral payload, not per-tool structured schemas) and the raw provider output it derived from; consumers choose which to read.
- Exposing raw provider output on Agent Events intentionally supersedes the earlier rule that live output hides raw provider stdout/JSON; raw is now a carried payload, while provider DTO objects, command rendering, and stream parsing remain internal.
- The finished-run log and Live Runtime Output share one Agent Event vocabulary: live emits events incrementally, the finished-run log is their complete ordered sequence plus terminal metadata; runtime owns no durable storage of either.
- Live Runtime Output is independent of session lifecycle and `ToolPolicy`; completed runtime output remains authoritative.
- Live Runtime Output observers are synchronous, notification-only, at-most-once per provider attempt, and consumer-owned for async bridging and backpressure.
- Live Runtime Output callback failures propagate as exceptional consumer failures.
- Session continuity and tool policy are independent runtime concerns.
- Ephemeral execution does not intentionally prepare provider-session continuity.
- Start Session Run returns continuation state; callers own persistence and retention.
- Resume Session Run continues an existing continuity chain without Consumer Fallback or reselection.
- Continuations are opaque, portable, semantically immutable, and may carry provider-owned serializable resume state.
- Continuations and invocation records must not intentionally persist ProviderAuth values.
- Runtime does not guarantee redaction of arbitrary prompt text, provider output, or diagnostics; consumers own durable redaction policy.
- Continuations may carry provider identity and resume state, but not ProviderSelection objects.
- Session-backed execution is limited to built-in providers that produce and consume portable continuation data.
- Runtime must not own durable provider-session storage, durable invocation-log storage, or cleanup policy.
- All resume state round-trips through the opaque Continuation: per session run the runtime may capture provider on-disk session state into the continuation and restore it into an ephemeral, self-cleaned working directory; it keeps no cross-call SessionStore, session-id persistence, or durable state directory between calls.
- Resumed-session availability or usage-limit failures do not invalidate continuations and must not trigger automatic Consumer Fallback inside runtime.
- Session-backed interruptions report invocation progress so callers can choose retry or continuation prompts.
- Usage limits, cancellation, timeout, temporary unavailability, and confidently retryable provider failures are normal `RuntimeOutcome` values.
- Credential failures, runtime configuration errors, hard provider failures, adapter/protocol bugs, unclassified provider failures, invalid service references, and unexpected exceptions remain exceptional failures.
- Provider-reported usage belongs on runtime outcomes whenever observed, including interrupted outcomes.
- Cancellation and timeout outcomes report only usage observed before interruption; runtime performs no provider-specific post-kill usage recovery.
- `timed_out` is an Idle Timeout: a heartbeat watchdog reset by every observed Agent Event, not a total wall-clock budget.
- Idle Timeout length is a consumer-supplied per-run parameter on every lifecycle entrypoint, default 300s.
- Runtime performs no automatic timeout retry or restart; it reports `timed_out` and the consumer decides whether to start a new invocation.
- A new continuation becomes meaningful only after provider work has started.
- Resumed-session execution keeps service, model, effort, and `ToolPolicy` fixed from the continuation while receiving credentials separately.
- Runtime requests require explicit `ToolPolicy`; non-`NONE` policies grant Invocation Directory as Tool Workspace.
- `ToolPolicy` is an invocation permission grant, not part of ProviderSelection identity.
- Invocation Directory and Tool Workspace are distinct permission concepts, but not separate public paths in the current runtime model.
- `ProviderSelection` is the canonical single-candidate selection value for one invocation.
- Runtime Compatibility Aliases are not Runtime Public Surface promises and may be removed before release without tailored migration behavior.
- Lifecycle-specific runtime execution adapter names are canonical public spellings even when they share adapter protocols.
- Live Provider Smoke Tests are opt-in maintainer tooling, not default automated tests or Runtime Public Surface additions.
- Live Provider Smoke Tests prove real provider invocation through Runtime Public Surface; they do not judge answer quality, tool usefulness, or strict instruction following.
- Live Smoke Defaults prefer the cheapest runtime-supported provider tuple over stronger models; smoke prompts must stay simple enough for those defaults.

## Flagged Ambiguities

- "Resumable": use **Start Session Run** and **Resume Session Run** for lifecycle APIs; keep resumable only for provider capabilities or lower-level planning.
- "One-shot": use **Ephemeral Run** for execution without provider-session continuity.
- "stage", "StageSelection", "StageOverride", "stage chain": use **ProviderSelection** and the `provider_selection` request field for one invocation, or **Consumer Fallback** for consuming-project retry orchestration.
- "OpenAI" in issue #93 meant **OpenCode**.
- "OpenCode subscription": use **OpenCode Go Integration** for the built-in `opencode` service; OpenCode Zen/pay-as-you-go models are outside that service.
- "Claude API key": use **ClaudeCodeOAuthToken**, not generic Anthropic API key.
- "Adapter author": custom provider services are not a supported runtime extension point.
- "worktree": use **Invocation Directory** for command location and **Tool Workspace** for tool access to that directory.
- `ToolAccess`: use **ToolPolicy**.
- `ToolPolicyProfile`: use **ToolPolicy** in consumer-facing language; provider flag profiles are internal.
- `InvocationRole`, `UsageLimitScope`, `SessionNamespace`: caller labels and grouping policy belong outside the Runtime Consumer Surface.
- `ToolPolicy.RESTRICTED`, `ToolPolicy.PARTIAL`, `ToolPolicy.FULL`: use `ToolPolicy.INSPECT_ONLY`, `ToolPolicy.NO_FILE_MUTATION`, `ToolPolicy.UNRESTRICTED`.
