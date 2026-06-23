# Live runtime output observation

Status: Superseded by 0012. The message-only, no-raw-output decision below is retained for history; see 0012 for the typed `Agent Event` model that carries raw output and is shared with the finished-run log.

`agent_runtime` needs a consumer-facing way to observe live agent-message output during provider execution. Pycastle uses provider-parsed assistant turns for live display and protocol handling; runtime must support that without exposing provider JSON events, raw stdout, or provider adapter seams as public API.

## Decision

- Add Live Runtime Output as a provider-neutral observation channel for agent-message turns emitted during runtime invocations.
- Live Runtime Output belongs to the Runtime Consumer Surface.
- `AgentMessageTurn` is shared public runtime vocabulary and may be exported from both package root and `agent_runtime.runtime`.
- Observers receive `AgentMessageTurn` values directly, not raw strings, speculative event wrappers, raw provider lines, or provider-specific event DTOs.
- Values include selected service identity, not model or effort metadata unless a separate live-display need appears.
- Observed turns come from one selected provider invocation; consumers that perform fallback across separate runtime calls own cross-invocation correlation.
- Observers should see each runtime-observed `AgentMessageTurn` at most once per provider attempt.
- Observation is per request through Live Runtime Output vocabulary such as `on_live_output`, not `RuntimeClient` display state or alternate streaming entrypoints.
- Observers are synchronous, notification-only, and do not steer runtime control flow; consumers bridge async queues, own backpressure, and cancel to stop invocation.
- Live Runtime Output applies across Ephemeral Run, Start Session Run, and Resume Session Run.
- Live Runtime Output reports only newly observed current-invocation output, not prior turns from continuations, logs, or provider transcript state.
- Live Runtime Output is not coupled to provider-session id detection.
- Completed runtime output remains authoritative; live observations do not change `RuntimeOutcome` semantics.
- Application protocol parsing of Live Runtime Output remains outside the runtime boundary.
- Consumers own display, redaction, and persistence policy for observed live output.
- Observed turns use the same runtime-owned provider parsing semantics as final output reduction, but are not guaranteed literal substrings of completed output.
- Observation is independent of `ToolPolicy` and grants no tool capability, workspace access, or logging permission.
- Built-in providers emit Live Runtime Output when runtime parsing observes an `AgentMessageTurn` equivalent; providers with no live turns still accept observers.
- Observer callback failures are consumer-side failures and propagate as exceptional consumer failures, not runtime interruption outcomes.

## Consequences

- Pycastle can migrate live `AssistantTurn` handling without constructing provider services or parsing provider streams.
- Runtime keeps provider stream parsing and DTOs internal while exposing the needed live behavior.
- Consumers that do not need live observation keep using lifecycle entrypoints exactly as before.
- Future provider-neutral live output kinds require new public-surface decisions.
