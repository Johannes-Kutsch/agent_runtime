# Live runtime output observation

Status: Accepted.

`agent_runtime` needs a consumer-facing way to observe live agent-message output during provider execution. Pycastle is a target consumer and currently uses provider-parsed assistant turns for live display and protocol handling. The runtime must support that use case without exposing provider-specific JSON events, raw stdout, or provider adapter seams as public API.

## Decision

- Add Live Runtime Output as a provider-neutral observation channel for agent-message turns emitted during runtime invocations.
- Live Runtime Output belongs to the Runtime Consumer Surface so ordinary consuming projects can observe agent-message turns without importing provider adapters or provider event parsers.
- `AgentMessageTurn` is shared public runtime vocabulary and may be exported from both the package root and `agent_runtime.runtime`.
- Live Runtime Output observers receive `AgentMessageTurn` values directly, modeled after pycastle's assistant-turn pattern, not raw strings, speculative event wrappers, raw provider lines, or provider-specific event DTOs.
- Live Runtime Output values include selected service identity so consumers can correlate observed turns during fallback or display.
- Live Runtime Output values do not repeat selected model or effort metadata unless a separate live-display need is established.
- Live Runtime Output may include turns from provider attempts later abandoned by fallback; consumers use service identity and authoritative final output to correlate or discard those observations.
- Live Runtime Output observers should see each runtime-observed `AgentMessageTurn` at most once per provider attempt.
- Observation is supplied per request, not as `RuntimeClient` display state and not through alternate streaming lifecycle entrypoints.
- The public per-request observer uses Live Runtime Output vocabulary, such as `on_live_output`, rather than provider- or parser-specific naming.
- Live Runtime Output observers are synchronous; async consumers bridge observation into their own queues or event loops outside the runtime boundary.
- Backpressure from synchronous Live Runtime Output observers is consumer responsibility; the runtime does not own buffering or drop policy for observed turns.
- Live Runtime Output observers are notification-only and do not steer runtime control flow; consumers use cancellation to stop an invocation.
- Live Runtime Output is available across Ephemeral Run, Start Session Run, and Resume Session Run because observing agent-message turns is independent of session lifecycle.
- Live Runtime Output reports only newly observed output from the current invocation and does not replay prior turns from continuations, logs, or provider transcript state.
- Live Runtime Output is not coupled to provider-session id detection and should not delay observed turns until session metadata is available.
- Live Runtime Output preserves provider-neutral agent-message turn boundaries rather than exposing arbitrary provider output chunks.
- Completed runtime output remains authoritative. Live Runtime Output does not change completed or interrupted `RuntimeOutcome` output semantics.
- Application-specific protocol parsing of Live Runtime Output remains outside the runtime boundary.
- Consumers own display, redaction, and persistence policy for observed live output.
- Live Runtime Output uses the same runtime-owned provider parsing semantics as final output reduction, but observed turns are not guaranteed to be literal substrings of authoritative completed output.
- Live Runtime Output observation is independent of `ToolPolicy` and grants no tool capability, workspace access, or logging permission.
- Built-in Provider Adapters should emit Live Runtime Output when runtime-owned provider parsing observes an `AgentMessageTurn` equivalent. A provider that emits no live turns must not reject the invocation solely because an observer is present.
- Observer callback failures are consumer-side failures, must not be silently swallowed by the runtime, and propagate as exceptional consumer failures rather than runtime interruption outcomes.

## Consequences

- Pycastle can migrate its live `AssistantTurn` handling to `agent_runtime` without constructing provider services or parsing provider streams itself.
- The runtime keeps provider stream parsing and provider DTOs internal while still exposing the specific live behavior consuming projects need.
- Consumers that do not need live observation can continue using lifecycle entrypoints exactly as before.
- Future provider-neutral live output kinds require new public-surface decisions rather than leaking provider-native event types through this channel.
