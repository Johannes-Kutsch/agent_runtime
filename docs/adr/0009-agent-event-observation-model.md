# Agent Event observation model

Consolidates the typed Agent Event decision and the live-feed-only / outcome-collapse decision.

Live Runtime Output (`on_live_output`) emitting typed `Agent Event`s is the sole output-observation channel. No stored, compiled, or finished-run log. Consumers wanting the whole run buffer the live feed themselves.

## Agent Event

- Closed type: agent message, agent tool call, other agent life sign. Consumers branch on type.
- Three fields: `type`, human-readable `display_message`, `raw_provider_output`. Structured detail (e.g. tool identity) available only by reading raw.
- Provider stream parser maps every consumed chunk to exactly one event; "other" catches non-message/non-tool-call chunks so raw output is never dropped and the full raw stream is reconstructable.
- Exposing raw provider output intentionally supersedes the earlier rule hiding raw stdout/JSON. Provider DTO objects, command rendering, stream-parsing internals stay internal.

## RuntimeOutcome and RunResult

- `RuntimeOutcome` is `{ kind, result }`. `kind` is a discriminated union: `Completed`, `UsageLimited(reset_time)`, `ProviderUnavailable(reason)`, `Cancelled`, `TimedOut`. Bad credentials stay an exception (`AgentCredentialFailureError`).
- `RunResult` always present, even after interruption: `output`, `usage`, `continuation` (none for ephemeral), `selected`.
- `ResolvedProvider` `{ service, model, effort }` — credential-free, distinct from `ProviderSelection`. Canonical wherever that triple appears.
- Callers infer resumability from continuation presence; `invocation_progress` is internal.

## Observation invariants

- Per request through `on_live_output`; observers synchronous, notification-only, at-most-once per provider attempt, no runtime control-flow steering.
- Current invocation only — no replay from continuations, logs, or transcripts.
- Independent of `ToolPolicy` and session lifecycle; grants no tool capability.
- `RuntimeOutcome` semantics authoritative. Observer callback failures propagate as exceptional consumer failures.

## Consequences

- One observation model learned once; no second at-rest vocabulary.
- Consumers wanting tool structure parse `raw_provider_output`; raw payloads may carry provider-shaped text consumers must redact.
- Outcome is a sum type: no `None`-by-default fields, consumers `match` on `kind` for terminus detail, common facts read off `result`.
- Runtime keeps provider DTOs, command rendering, stream parsing internal.
