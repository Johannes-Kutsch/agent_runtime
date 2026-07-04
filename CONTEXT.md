# agent_runtime Context

## Purpose

`agent_runtime` is the reusable runtime boundary for agent execution. It owns contracts consumed by application adapters without importing the application.

## Ubiquitous Language

| Term | Meaning |
| --- | --- |
| `agent_runtime` | Reusable runtime package and stable core public surface. |
| `Runtime Public Surface` | Documented stability surface: consumer entrypoints, runtime value objects, built-in provider selection — not every importable symbol. |
| `Runtime Consumer Surface` | Consuming-project entrypoints for executing prepared agent work without implementing runtime or provider adapters. |
| `Built-in Provider Rendering` | Internal in-process module behind `RuntimeClient` that turns normalized provider selection facts, run kind, `ToolAccess`, `ProviderAuth`, Invocation Directory, optional provider state location, optional provider-session id, and needed host process facts into a rendered built-in provider invocation: canonical argv, legacy command text where still needed, environment, prompt path, prompt cleanup choice, prompt transport preference, and provider-session id placement. It owns built-in provider model and effort allowlists, provider-specific command and environment rendering, prompt transport decisions, `ToolPolicy` mapping, OpenCode config generation, and provider auth validation, but performs no subprocess I/O, no provider stream interpretation, and no Windows host-process environment layering (now owned by Built-in Provider Invocation). |
| `Built-in Provider Invocation` | Internal mechanism behind `RuntimeClient` executing one prepared built-in provider command and returning observable invocation facts. Layers the Windows host-process environment allowlist onto every invocation's environment once (provider values take precedence), so no render path can omit it. |
| `Built-in Provider Stream Interpretation` | Internal in-process module behind `RuntimeClient` that interprets merged built-in provider output lines into final output, `ProviderUsage`, provider-session identifiers, started/not-started facts, and `Agent Event` values delegated through Built-in Provider Agent Event Building. It owns Claude, Codex, and OpenCode stream semantics but performs no provider subprocess I/O. |
| `Built-in Provider Agent Event Building` | Internal in-process module behind Built-in Provider Stream Interpretation that turns one raw built-in provider output line into one Agent Event for Claude, Codex, and OpenCode. It owns provider-specific live event construction, including JSON decoding, non-object and raw-text fallback, message events, tool-call events, turn-summary display formatting, provider-specific other descriptors, and raw provider output preservation. It performs no provider subprocess I/O, output reduction, usage extraction, provider failure classification, provider-session id extraction, or invocation-progress classification. |
| `ProviderAuth` | Immutable credential data carried by `ProviderSelection`, or supplied to Resume Session Run since continuations don't store credentials. |
| `ProviderSelection` | Public request value selecting one service, model, effort, and provider credentials for one invocation. |
| `OpenCode Go Integration` | Built-in `opencode` service integration for OpenCode Go subscription access. |
| `Consumer Fallback` | Consuming-project orchestration choosing whether to start a separate runtime invocation after a prior one completes or fails. |
| `ToolPolicy` | Closed public value: `NONE`, `NO_FILE_MUTATION`, or `UNRESTRICTED`. |
| `Invocation Directory` | Host directory where runtime launches a provider command; request field `invocation_dir`. |
| `Tool Workspace` | Invocation Directory when runtime exposes it through provider tools. |
| `Idle Timeout` | Per-run liveness watchdog owned by Built-in Provider Invocation: max seconds without any raw provider output line before the runtime terminates the provider subprocess and yields a `timed_out` outcome. Reset by any raw output line, not by interpreted `Agent Event`s. Consumer-configurable; default 300s. |
| `CancellationToken` | Consumer-owned cooperative cancellation handle passed via a lifecycle request's `token` field. `.cancel()` is safe to call from a different thread than the one blocked executing the invocation — this is the expected usage, not an edge case, since a run occupies whatever thread awaits it. Checked once before the provider subprocess spawns (skips spawning if already cancelled) and polled sub-second during execution; takes precedence over Idle Timeout if both would fire together. Terminates the provider subprocess via the same hard-kill path as Idle Timeout, no graceful-shutdown attempt. Preserves a `Continuation` only when provider work had started, mirroring Idle Timeout's continuation-preservation rule. |
| `Ephemeral Run` | Invocation that neither prepares nor promises provider-session continuity. |
| `Start Session Run` | Invocation that selects a service and prepares provider-session continuity. |
| `Resume Session Run` | Invocation that continues an existing continuity chain without fallback or reselection. |
| `Session-backed Provider` | Built-in provider that can produce and consume portable continuation data. |
| `Session-backed Provider Execution` | Internal lifecycle module behind `RuntimeClient` Start Session Run and Resume Session Run for `Session-backed Provider`s. It asks Session-backed Provider State Resolution for provider-state facts, invokes through `Built-in Provider Invocation`, interprets expected interruption, builds `RuntimeOutcome` / `RunResult` values, resolves observed provider-session identity from invocation output, and preserves `Continuation` only when provider work started and a provider-session id is known. |
| `Session-backed Provider State Resolution` | Private internal module used by Session-backed Provider Execution that owns provider-state filesystem facts and continuation-preparation facts for Claude, Codex, and OpenCode. It prepares/restores provider homes in the caller-owned Session Store, computes provider-state relative pointers, provider-state resumability, prepared or recovered provider-session id facts, `RunKind`, exact-transcript-match facts, and provider-specific continuation input facts. It performs no provider subprocess I/O, no provider output parsing, no provider-session id observation from invocation output, and no `RuntimeOutcome` / `RunResult` construction. |
| `Continuation` | Opaque portable resume token callers persist and pass back to resume a continuity chain. Carries resume identity and an optional relative-path pointer into the caller's `Session Store`; not the provider session bytes themselves. An empty pointer means the Store root itself is the provider home (the current layout for new continuations); a non-empty pointer resolves a subpath (legacy tokens written under the retired `implementer/<provider>` layout). |
| `Session Store` | Caller-owned, isolated directory that **is** a session-backed provider's isolated home, holding its native on-disk session state directly at the Store root. Supplied symmetrically to `Start Session Run` and `Resume Session Run` so a `Continuation` can resume. The runtime points the provider's home environment at the Store root and reads/writes within it, but never owns, creates durably, cleans, serializes, or recreates it. Holds one provider's home per Store; distinct from the host user's native provider home. |
| `Live Runtime Output` | Live feed of typed `Agent Event` observations emitted during invocation; the runtime's only output-observation channel. |
| `Agent Event` | One observed signal in Live Runtime Output: closed type (agent message, agent tool call, turn summary, or other agent life sign), single human-readable display message, raw provider output it derived from. |
| `RuntimeClient` | Caller-owned runtime object executing requests without durable provider session storage or cross-call availability policy. |
| `Execution Argv Transform` | Optional per-invocation callable on run request objects (`EphemeralRunRequest`, `NewSessionRunRequest`, `ResumedSessionRunRequest`) that transforms the fully-rendered canonical argv, Invocation Directory, and rendered environment into a new argv before Built-in Provider Invocation executes it. Enables consumers to route provider CLI execution to non-host environments (e.g. Docker containers) without ar gaining a container dependency. ar retains full subprocess execution ownership; the transform is a pure synchronous data transformation. When present on a Codex invocation, Built-in Provider Rendering automatically applies `--sandbox danger-full-access` regardless of `ToolPolicy`, because a custom transform implies a non-standard execution environment where Codex's OS sandbox cannot be assumed to work. |
| `RuntimeOutcome` | One invocation's outcome: discriminated `kind` plus `RunResult`. `kind` is closed — completion, usage limits, provider unavailability (with closed reason), cancellation, timeout; credential and hard failures are exceptions. |
| `Runtime Outcome Folding` | Internal in-process module behind `RuntimeClient` that converts completed `RunResult`s and expected interruption exceptions into caller-facing `RuntimeOutcome` values for Ephemeral Run, Start Session Run, and Resume Session Run. It preserves live-output callback failures and lets credential, hard provider, configuration, and unexpected failures remain exceptional. It owns selected-provider fallback facts for interrupted outcomes but performs no provider invocation, stream interpretation, provider rendering, session-store mutation, or lifecycle-specific continuation construction. |
| `ProviderUnavailable` | RuntimeOutcome kind for temporary provider failures: closed reason (`SERVICE_NOT_AVAILABLE`, `TRANSIENT_API_ERROR`) plus raw detail string. |
| `RunResult` | Run facts carried by every `RuntimeOutcome`, even after interruption: final output, provider usage, resume continuation (none for ephemeral), `ResolvedProvider`. |
| `ResolvedProvider` | Credential-free identity of the provider actually run: service, model, effort. Distinct from `ProviderSelection`; canonical wherever that triple appears. |
| `Live Provider Probe` | Opt-in manual debugging tool (not CI, not default tests, not Runtime Public Surface) exercising real built-in providers; streams events live, writes per-case JSON artifacts wiped on rerun. |
| `Live Probe Case Runner` | Private Live Provider Probe module/interface that executes one planned probe case through the Runtime Public Surface, writes per-case live feed and result artifacts, classifies the outcome, captures traceback, and returns only facts needed by provider-level sequencing and terminal display. It is not Runtime Public Surface. |
| `Live Probe Case Matrix` | Per-service: three entry paths at `UNRESTRICTED`, plus ephemeral under each remaining `ToolPolicy` — five cases, deduplicated on `ephemeral_UNRESTRICTED`. |
| `Live Probe Default` | Cost-first runtime-supported provider/model/effort tuple used by the probe absent CLI override. |
| `ProviderUsage` | Provider-reported usage: input/output tokens, cache-read/cache-creation input tokens, optional USD cost, optional provider duration. |
| `ContinuationUnrecoverableError` | Exception raised when a session-backed run detects that the provider-side session state is gone despite a valid `Continuation` and `Session Store` being present — from either of two triggers: (1) a pre-flight resolver check on local session state, applied uniformly by all three session-backed providers, that hands control back to the consumer instead of silently starting a fresh session or reusing the continuation's id; or (2) Built-in Provider Stream Interpretation recognizing Claude's session-not-found CLI signal mid-invocation — the residual case where the pre-flight check passed but the session id does not exist in the provider home Claude actually used. The session-conflict ("already in use") CLI signal is deliberately **not** a trigger: it is structurally prevented (see boundary rules) and, if it ever surfaces, is a plain hard error. Carries `service_name`, plus optional `classification` (which known signal fired) and `raw_message` (the exact provider text) for diagnostics. Not a configuration error — the caller did nothing wrong. The consumer catches it, drops the stale continuation, and re-plans. |
| `ModelNotAvailable` | `RuntimeOutcome` kind signalling that the selected model is not available for the caller's account tier on the named service; the service itself and its other models remain accessible. Carries no fields — the rejected model identity is readable from `RunResult.resolved_provider`. Consumer Fallback uses this to retry with a different model on the same service. Distinct from `ProviderUnavailable` (service-level or transient failure) and `UsageLimited` (quota exhaustion). |

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
- Runtime Outcome Folding is internal and must not become a port, adapter, consumer extension point, or Runtime Public Surface; lifecycle callers pass execution and selected-provider facts instead of reimplementing interruption-to-outcome mapping.
- Built-in Provider Rendering sits above Built-in Provider Invocation: rendering produces invocation facts; invocation adapters execute those facts and deliver merged provider output for stream interpretation.
- Built-in Provider Rendering is internal and must not become a consumer extension point or Runtime Public Surface.
- Built-in provider model and effort allowlists, command and environment rendering, prompt path and cleanup choices, prompt transport preferences, OpenCode config generation, and provider-specific auth requirements belong in Built-in Provider Rendering.
- Windows host-process environment layering belongs in Built-in Provider Invocation, not rendering: the allowlist (`PATH`, `PATHEXT`, `SystemRoot`, `ComSpec`, `WINDIR`) is layered once at the single execution chokepoint, beneath provider-supplied values, so no render path can omit it. No-op on POSIX.
- Built-in Provider Invocation is internal and must not become a consumer extension point.
- Built-in Provider Stream Interpretation sits above Built-in Provider Invocation: invocation adapters execute provider processes and deliver the merged stdout/stderr line stream; stream interpretation owns provider-shaped semantics for Claude, Codex, and OpenCode.
- Built-in Provider Agent Event Building sits inside Built-in Provider Stream Interpretation: lifecycle code continues to consume stream interpretation and its `build_agent_event` callable; the event-building module is not a port, adapter, consumer extension point, or Runtime Public Surface.
- Built-in Provider Agent Event Building accepts one raw provider output line and returns one Agent Event; every consumed Live Runtime Output chunk preserves `raw_provider_output`.
- Output reduction, usage extraction, provider failure classification, provider-session id extraction, and invocation-progress classification stay in Built-in Provider Stream Interpretation or its existing deep modules, not in Built-in Provider Agent Event Building.
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
- Runtime must not own durable provider-session storage, invocation-log storage, or cleanup policy. The opaque Continuation round-trips resume identity plus a pointer into the caller-owned Session Store; the session's on-disk provider bytes live in the Session Store, which the consumer owns and persists. Resume requires both.
- The runtime never serializes, embeds, or recreates provider session state. Session-backed runs (Start Session Run, Resume Session Run) require a caller-owned Session Store, supplied symmetrically across new and resume; the runtime points the provider home at the Store root and lets the provider resolve its own session by id. Omitting the Store on a session-backed run is a configuration error. The provider's isolated home is the Store root directly — no runtime-invented owner/provider/namespace subdirectory. A Session-backed Provider's Continuation carries an optional relative-path pointer into the Store: empty for new continuations (Store root is the home), non-empty only for legacy tokens written under the retired `implementer/<provider>` layout, which resume still resolves by joining the stored pointer. The runtime invents no role or namespace segment; scoping a Store per session/role is the caller's responsibility, not the runtime's.
- Every built-in provider run executes against an isolated provider home, never the host user's native provider home. Ephemeral Runs use a throwaway scratch home; session-backed runs use the caller-owned Session Store. Provider credentials are seeded into the isolated home from the host.
- Resumed-session failures do not invalidate continuations or trigger automatic Consumer Fallback inside runtime.
- Session-backed interruptions surface a continuation only when provider work started; callers infer resumability from continuation presence.
- `Session-backed Provider Execution` and `Session-backed Provider State Resolution` are internal and must not become consumer extension points; lifecycle code reuses `Built-in Provider Invocation`, while provider-specific filesystem and continuation-preparation facts stay explicit inside provider-state resolution.
- `RuntimeClient` remains the `Runtime Public Surface` for Start Session Run and Resume Session Run; consumers never import the session-backed execution module.
- Credential failures, hard provider failures (including process-level: non-zero exit or empty output), adapter/protocol bugs, and unexpected exceptions remain exceptional.
- The runtime raises `ContinuationUnrecoverableError` — not `RuntimeConfigurationError` — whenever provider-side session state is confirmed gone despite a valid `Continuation`, from either of two trigger points: Session-backed Provider State Resolution's pre-flight check on local session state (valid token present, provider-side session state absent), applied uniformly across all three session-backed providers; or Built-in Provider Stream Interpretation recognizing Claude's session-not-found CLI signal during a Claude invocation (the residual case where the pre-flight check passed but the id is absent in the provider home Claude actually used). Consumers own the decision to drop the continuation and re-plan in both cases.
- A resume with absent local session state must raise `ContinuationUnrecoverableError` and hand control back to the consumer; it must never silently downgrade to a fresh start, and never reuse the continuation's session id for a fresh start. This holds for all three session-backed providers (Codex already did; Claude and OpenCode are brought into line).
- New sessions never start with a caller-influenced session id; the runtime mints a fresh, never-before-used id for each. A provider's "session id already in use" report is therefore structurally impossible from a correct runtime and is classified as a plain hard error, never `ContinuationUnrecoverableError`.
- Provider-reported usage belongs on outcomes whenever observed, including interrupted outcomes. Cancellation/timeout outcomes report only usage observed before interruption.
- Idle Timeout is a liveness watchdog reset by every raw provider output line, not a wall-clock budget. Runtime performs no automatic timeout retry.
- Resumed-session execution keeps service, model, effort, `ToolPolicy` fixed from continuation; credentials received separately.
- Runtime requests require explicit `ToolPolicy`; non-`NONE` policies grant Invocation Directory as Tool Workspace.
- `ToolPolicy` is an invocation permission grant, not part of ProviderSelection identity.
- `Execution Argv Transform` is a per-run request field, not a `RuntimeClient` constructor fact; a single client may dispatch invocations to different execution environments.
- When `Execution Argv Transform` is present, Built-in Provider Invocation applies it before host executable resolution, forces stdin prompt transport, and passes the full rendered environment as the third argument so the consumer can inject ar-generated values (e.g. `OPENCODE_CONFIG_CONTENT`) into the target environment.
- `Execution Argv Transform` must be a pure synchronous callable; ar owns all subprocess I/O, timeout, and exit-code handling.
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
- `UsageLimitScope`: outside Runtime Consumer Surface.
- `runtime_state_dir` / `_runtime_state_dir`: use **Session Store**; it is caller-owned and a public, required input for session-backed runs, not an internal field.
- `ToolPolicy.RESTRICTED`, `.PARTIAL`, `.FULL`: use `.NO_FILE_MUTATION`, `.UNRESTRICTED`.
- `ToolPolicy.INSPECT_ONLY`: retired; use `NO_FILE_MUTATION` (bash permitted, file mutations denied) or `NONE` (no tools).
- `RetryableProviderFailure`, `RetryableProviderFailureError`: use **ProviderUnavailable** / `ProviderUnavailableError`.
- `NoServiceAvailable`, `NoServiceAvailableError`: use **ProviderUnavailable** with reason `SERVICE_NOT_AVAILABLE`.
- `ProviderErrorObservation`, `observations`: retired; diagnostics use raw error messages only.
- `Live Runtime Output Timeout Context`, event-layer idle watchdog: retired; **Idle Timeout** now lives in Built-in Provider Invocation and resets on raw output lines.
