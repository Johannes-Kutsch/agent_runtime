# Typed Agent Events with raw output, shared by live and finished-run log

Status: Accepted, partially superseded by ADR 0012 (finished-run log / `InvocationRecord` removed, `Agent Event` collapsed to a single display message, live feed is the only channel). Supersedes the earlier message-only live-output decision (ADR 0007, since removed).

The earlier Live Runtime Output decision made it a message-only channel: observers received `AgentMessageTurn` values and explicitly never raw provider stdout, JSON, or provider DTOs. The intended runtime scope is broader. Consumers need to observe not just agent message text but also agent tool calls and other agent life signs (to tell an agent is alive and working while silent), and need the raw provider output available alongside the filtered view when the neutral view is insufficient. They also need the finished-run log and the live channel to share one vocabulary, rather than the live channel being typed turns while the at-rest record is a raw blob the consumer must re-parse.

## Decision

- Replace `AgentMessageTurn` with `Agent Event`: one closed type discriminated into agent message, agent tool call, and other agent life sign. Consumers branch on type.
- Each `Agent Event` carries both a filtered, provider-neutral view (message text; tool identity plus a neutral payload, not a per-tool structured schema; a neutral life-sign descriptor) and the raw provider output fragment it was derived from, plus selected service identity.
- The provider stream parser maps every consumed chunk to exactly one `Agent Event`; chunks that are neither message nor tool call become "other" events, so raw output is never dropped and the full raw stream is reconstructable by concatenation.
- Exposing raw provider output on `Agent Event`s intentionally supersedes the earlier rule that live output hides raw provider stdout/JSON. Raw is now a carried payload. Provider DTO objects, command rendering, and stream-parsing internals stay runtime-owned and internal.
- The finished-run log (`InvocationRecord`) and Live Runtime Output share this one `Agent Event` vocabulary: live emits events incrementally; the finished-run log is their complete ordered sequence plus terminal metadata (outcome category, usage, selected service/model/effort, provider session id, run kind) and raw evidence.
- The runtime owns no durable storage of either channel. It returns events and records as in-memory data; consumers own persistence, file layout, naming, redaction, retention, and cleanup. Durable invocation-log file writing is removed from the invocation path.
- Carried-over invariants from the earlier decision still hold: observation is per request through `on_live_output`; observers are synchronous, notification-only, at most once per provider attempt, and do not steer runtime control flow; events come from the current invocation only, with no replay of prior turns from continuations, logs, or transcript state; observation is independent of `ToolPolicy` and session lifecycle and grants no tool capability; completed runtime output and `RuntimeOutcome` semantics remain authoritative; observer callback failures propagate as exceptional consumer failures.

## Consequences

- One event model serves both "stream me the run" and "give me the whole run"; consumers learn a single vocabulary.
- Consumers gain tool-call and life-sign visibility and a raw fallback, at the cost of strict provider-neutrality; raw payloads may carry provider-shaped text consumers must treat as opaque and redact themselves.
- The runtime keeps provider DTOs, command rendering, and stream parsing internal while exposing richer observable behavior.
- Removing durable log writing means consumers that relied on runtime-written files must persist returned records themselves.
- Adding further `Agent Event` types later remains a public-surface change but no longer requires re-deciding whether raw output may be exposed.
