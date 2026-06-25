# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts consumed by application adapters without importing the application.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | Reusable runtime package and stable core public surface. |
| `Runtime Public Surface` | Documented stability surface: consumer entrypoints, runtime value objects, built-in provider selection — not every importable symbol. |
| `Runtime Consumer Surface` | Consuming-project entrypoints for executing prepared agent work without implementing runtime or provider adapters. |
| `Built-in Provider Rendering` | Internal in-process module behind `RuntimeClient` that turns normalized provider selection facts, run kind, `ToolAccess`, `ProviderAuth`, Invocation Directory, optional provider state location, optional provider-session id, and needed host process facts into a rendered built-in provider invocation: canonical argv, legacy command text where still needed, environment, prompt path, prompt cleanup choice, prompt transport preference, and provider-session id placement. It owns built-in provider model and effort allowlists, provider-specific command and environment rendering, prompt transport decisions, `ToolPolicy` mapping, Windows process environment allowlisting, OpenCode config generation, and provider auth validation, but performs no subprocess I/O and no provider stream interpretation. |
| `Built-in Provider Invocation` | Internal mechanism behind `RuntimeClient` executing one prepared built-in provider command and returning observable invocation facts. |
| `Built-in Provider Stream Interpretation` | Internal in-process module behind `RuntimeClient` that interprets merged built-in provider output lines into `Agent Event` values, final output, `ProviderUsage`, provider-session identifiers, and started/not-started facts. It owns Claude, Codex, and OpenCode stream semantics but performs no provider subprocess I/O. |
| `ProviderAuth` | Immutable credential data carried by `ProviderSelection`, or supplied to Resume Session Run since continuations don't store credentials. |
| `ProviderSelection` | Public request value selecting one service, model, effort, and provider credentials for one invocation. |
| `OpenCode Go Integration` | Built-in `opencode` service integration for OpenCode Go subscription access. |
| `Consumer Fallback` | Consuming-project orchestration choosing whether to start a separate runtime invocation after a prior one completes or fails. |
| `ToolPolicy` | Closed public value: `NONE`, `INSPECT_ONLY`, `NO_FILE_MUTATION`, or `UNRESTRICTED`. |
| `Invocation Directory` | Host directory where runtime launches a provider command; request field `invocation_dir`. |
| `Tool Workspace` | Invocation Directory when runtime exposes it through provider tools. |
| `Idle Timeout` | Per-run liveness watchdog owned by Built-in Provider Invocation: max seconds without any raw provider output line before the runtime terminates the provider subprocess and yields a `timed_out` outcome. Reset by any raw output line, not by interpreted `Agent Event`s. Consumer-configurable; default 300s. |
| `Ephemeral Run` | Invocation that neither prepares nor promises provider-session continuity. |
| `Start Session Run` | Invocation that selects a service and prepares provider-session continuity. |
| `Resume Session Run` | Invocation that continues an existing continuity chain without fallback or reselection. |
| `Session-backed Provider` | Built-in provider that can produce and consume portable continuation data. |
| `Session-backed Provider Execution` | Internal lifecycle module behind `RuntimeClient` Start Session Run and Resume Session Run for `Session-backed Provider`s. It resolves provider state, chooses run kind, invokes through `Built-in Provider Invocation`, interprets expected interruption, builds `RuntimeOutcome` / `RunResult` values, and preserves `Continuation` only when provider work started and a provider-session id is known. |
| `Continuation` | Opaque portable resume token callers persist and pass back to resume a continuity chain. |
| `Live Runtime Output` | Live feed of typed `Agent Event` observations emitted during invocation; the runtime's only output-observation channel. |
| `Agent Event` | One observed signal in Live Runtime Output: closed type (agent message, agent tool call, other agent life sign), single human-readable display message, raw provider output it derived from. |
| `RuntimeClient` | Caller-owned runtime object executing requests without durable provider session storage or cross-call availability policy. |
| `RuntimeOutcome` | One invocation's outcome: discriminated `kind` plus `RunResult`. `kind` is closed — completion, usage limits, provider unavailability (with closed reason), cancellation, timeout; credential and hard failures are exceptions. |
| `ProviderUnavailable` | RuntimeOutcome kind for temporary provider failures: closed reason (`SERVICE_NOT_AVAILABLE`, `TRANSIENT_API_ERROR`) plus raw detail string. |
| `RunResult` | Run facts carried by every `RuntimeOutcome`, even after interruption: final output, provider usage, resume continuation (none for ephemeral), `ResolvedProvider`. |
| `ResolvedProvider` | Credential-free identity of the provider actually run: service, model, effort. Distinct from `ProviderSelection`; canonical wherever that triple appears. |
| `Live Provider Probe` | Opt-in manual debugging tool (not CI, not default tests, not Runtime Public Surface) exercising real built-in providers; streams events live, writes per-case JSON artifacts wiped on rerun. |
| `Live Probe Case Runner` | Private Live Provider Probe module/interface that executes one planned probe case through the Runtime Public Surface, writes per-case live feed and result artifacts, classifies the outcome, captures traceback, and returns only facts needed by provider-level sequencing and terminal display. It is not Runtime Public Surface. |
| `Live Probe Case Matrix` | Per-service: three entry paths at `UNRESTRICTED`, plus ephemeral under each remaining `ToolPolicy` — six cases, deduplicated on `ephemeral_UNRESTRICTED`. |
| `Live Probe Default` | Cost-first runtime-supported provider/model/effort tuple used by the probe absent CLI override. |
| `ProviderUsage` | Provider-reported usage: input/output tokens, cache-read/cache-creation input tokens, optional USD cost, optional provider duration. |

## Boundary Rules

- Runtime package must import without application modules.
- Application prompt rendering, CLI wiring, issue orchestration, output parsing, display, redaction, durable trace persistence, and retention belong outside runtime.
- Runtime Consumer Surface uses `RuntimeClient`, lifecycle requests, public outcome/result values, `ProviderSelection`, `ProviderAuth`, `Continuation`, and `ToolPolicy`.
- Consumers import `ToolPolicy` from public runtime/root modules, not adapter contract modules.
- Ordinary consumers select built-in providers through runtime call arguments, not provider services, registries, adapters, command builders, or DTO streams.
- Consumer-defined provider services are not supported.
- `RuntimeClient` owns no cross-call provider availability policy, durable provider state, or logs.
- Provider selection is caller-supplied via `provider_selection`; runtime validates built-in service, model, effort, and credentials.
- `ProviderSelection` requires explicit service, model, effort; selection defaults belong to consumers or maintainer tooling.
- OpenCode Go Integration accepts service-local model ids; provider-specific prefixes are internal rendering details.
- `ProviderSelection.auth` is optional; providers requiring explicit credentials validate during invocation.
- `ProviderSelection` equality includes credentials, but textual representations must not reveal credential values. `ProviderAuth` equality includes credential values, but textual representations must redact them.
- Resume Session Run derives provider identity from the continuation; accepts request-time ProviderAuth only for credentials.
- Runtime performs no provider fallback inside a single invocation; fallback is Consumer Fallback across separate invocations.
- Runtime outcomes and exceptions describe one invocation only and do not classify Consumer Fallback eligibility.
- `ProviderUnavailable` describes temporary unavailability, not exhaustion of a provider chain. Unsupported selections are configuration errors.
- Runtime exposes no finished-run log or stored Agent Event sequence; the only observation is the live feed.
- Missing explicit credentials are credential failures and do not trigger Consumer Fallback inside runtime.
- Runtime classifies provider failures but never acts on the classification: no waiting, retry, or in-runtime fallback. Surfaced outcomes carry a closed reason plus raw provider error.
- Expected provider failures are normal return values; hard errors, credential failures, process-level failures, and unclassified failures remain exceptions.
- Built-in Provider Rendering sits above Built-in Provider Invocation: rendering produces invocation facts; invocation adapters execute those facts and deliver merged provider output for stream interpretation.
- Built-in Provider Rendering is internal and must not become a consumer extension point or Runtime Public Surface.
- Built-in provider model and effort allowlists, command and environment rendering, prompt path and cleanup choices, prompt transport preferences, Windows process environment allowlisting, OpenCode config generation, and provider-specific auth requirements belong in Built-in Provider Rendering.
- Built-in Provider Invocation is internal and must not become a consumer extension point.
- Built-in Provider Stream Interpretation sits above Built-in Provider Invocation: invocation adapters execute provider processes and deliver the merged stdout/stderr line stream; stream interpretation owns provider-shaped semantics for Claude, Codex, and OpenCode.
- WorkInvocation and execution adapter seams are internal; public-looking internal modules should be underscore-prefixed before release.
- Retiring a concept means deleting obsolete code; internal survival requires a current built-in purpose and must not reappear on public surfaces.
- Built-in Provider Invocation must use runtime-neutral internal names, not pycastle-specific naming.
- Provider event DTOs, session details, command rendering, stream interpretation, and flag profiles are internal.
- Provider failure diagnostics carry raw error messages; structured provider error observations are not a runtime concept.
- Raw provider output on Agent Events supersedes the earlier rule hiding raw stdout/JSON; provider DTOs, command rendering, and stream parsing remain internal.
- Live Runtime Output: sole channel, events emitted incrementally and never accumulated or stored.
- Live Runtime Output independent of session lifecycle and `ToolPolicy`; completed output remains authoritative.
- Live Runtime Output observers are synchronous, notification-only, at-most-once per provider attempt; consumer-owned for async bridging. Callback failures propagate as exceptional failures.
- Idle Timeout lives in Built-in Provider Invocation because terminating a silent subprocess requires the process handle; it reads provider output against a deadline, resets on any raw output line, and kills the process on silence. The Live Runtime Output observer carries no timeout responsibility.
- Continuations are opaque, portable, semantically immutable; may carry provider-owned resume state but not ProviderSelection objects or ProviderAuth values (`ResolvedProvider` is credential-free).
- Runtime does not guarantee redaction of prompt text, provider output, or diagnostics; consumers own durable redaction policy.
- Runtime must not own durable provider-session storage, invocation-log storage, or cleanup policy. All resume state round-trips through the Continuation.
- Resumed-session failures do not invalidate continuations or trigger automatic Consumer Fallback inside runtime.
- Session-backed interruptions surface a continuation only when provider work started; callers infer resumability from continuation presence.
- `Session-backed Provider Execution` is internal and must not become a consumer extension point; it reuses `Built-in Provider Invocation` and keeps provider-specific continuity facts explicit inside the implementation.
- `RuntimeClient` remains the `Runtime Public Surface` for Start Session Run and Resume Session Run; consumers never import the session-backed execution module.
- Credential failures, hard provider failures (including process-level: non-zero exit or empty output), adapter/protocol bugs, and unexpected exceptions remain exceptional.
- Provider-reported usage belongs on outcomes whenever observed, including interrupted outcomes. Cancellation/timeout outcomes report only usage observed before interruption.
- Idle Timeout is a liveness watchdog reset by every raw provider output line, not a wall-clock budget. Runtime performs no automatic timeout retry.
- Resumed-session execution keeps service, model, effort, `ToolPolicy` fixed from continuation; credentials received separately.
- Runtime requests require explicit `ToolPolicy`; non-`NONE` policies grant Invocation Directory as Tool Workspace.
- `ToolPolicy` is an invocation permission grant, not part of ProviderSelection identity.
- `ProviderSelection` is the canonical single-candidate selection value for one invocation.
- Runtime Compatibility Aliases are not Runtime Public Surface promises and may be removed before release.
- Live Provider Probe is opt-in manual debugging — not CI, default tests, or Runtime Public Surface. Proves invocation, not answer quality or tool usefulness. Live Probe Defaults prefer cheapest runtime-supported tuple; prompts must stay simple enough for those defaults.
- Live Probe Case Runner is private to the manual-debug probe; it must not become Runtime Public Surface or change the probe artifact schema, CLI behavior, exit behavior, or ADR 0010 posture.

## Flagged Ambiguities

- "Resumable": use **Start Session Run** / **Resume Session Run** for lifecycle APIs; keep resumable only for provider capabilities.
- "One-shot": use **Ephemeral Run**.
- "stage", "StageSelection", "StageOverride", "stage chain": use **ProviderSelection** / **Consumer Fallback**.
- "OpenAI" in issue #93 meant **OpenCode**.
- "OpenCode subscription": use **OpenCode Go Integration**; OpenCode Zen/pay-as-you-go models are outside that service.
- "Claude API key": use **ClaudeCodeOAuthToken**.
- "Adapter author": custom provider services are not a supported extension point.
- "worktree": use **Invocation Directory** / **Tool Workspace**.
- `ToolAccess`: use **ToolPolicy**.
- `ToolPolicyProfile`: use **ToolPolicy**; provider flag profiles are internal.
- `InvocationRole`, `UsageLimitScope`, `SessionNamespace`: outside Runtime Consumer Surface.
- `ToolPolicy.RESTRICTED`, `.PARTIAL`, `.FULL`: use `.INSPECT_ONLY`, `.NO_FILE_MUTATION`, `.UNRESTRICTED`.
- `RetryableProviderFailure`, `RetryableProviderFailureError`: use **ProviderUnavailable** / `ProviderUnavailableError`.
- `NoServiceAvailable`, `NoServiceAvailableError`: use **ProviderUnavailable** with reason `SERVICE_NOT_AVAILABLE`.
- `ProviderErrorObservation`, `observations`: retired; diagnostics use raw error messages only.
- `Live Runtime Output Timeout Context`, event-layer idle watchdog: retired; **Idle Timeout** now lives in Built-in Provider Invocation and resets on raw output lines.
